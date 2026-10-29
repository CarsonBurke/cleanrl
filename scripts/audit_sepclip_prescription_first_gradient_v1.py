"""First recorded full clipped gradient versus source-reconstructed PPO/PG.

Private CUDA saved-shuffle reconstruction; actual source loss/clip must match
every recorded actor/critic gradient BITWISE before any covector comparison.
Raw gradients and LS scale are reconstructed, not independently saved PG.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.sepclip_prescription_replay import MODULES,load_replay,require_pin,sha256,validate_provenance


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):raise ValueError('Nonfinite first-gradient bridge array')


def tensor_bytes_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def exact_first_permutation(module,args,saved_shuffle,device):
    if not isinstance(saved_shuffle,torch.Tensor) or saved_shuffle.dtype!=torch.uint8 or saved_shuffle.ndim!=1 or saved_shuffle.numel()==0:
        raise ValueError('Require independently saved shuffle RNG byte state')
    if args.num_minibatches!=1 or args.minibatch_size!=args.batch_size:
        raise ValueError('Require one complete recorded full minibatch')
    generator=torch.Generator(device=device)
    try:generator.set_state(saved_shuffle.detach().cpu().clone())
    except RuntimeError as error:raise ValueError('Saved shuffle RNG state is not compatible with actual CUDA generator') from error
    rows=module.device_minibatches(args.batch_size,args.minibatch_size,device,generator)
    if len(rows)!=1 or rows[0].numel()!=args.batch_size:
        raise ValueError('Source minibatch routine did not produce complete first permutation')
    indices=rows[0].clone()
    expected=torch.arange(args.batch_size,device=device,dtype=indices.dtype)
    if not torch.equal(indices.sort().values,expected):raise ValueError('Incomplete first batch permutation')
    return indices,dict(saved_rng_sha256=tensor_bytes_hash(saved_shuffle),
        first_permutation_sha256=tensor_bytes_hash(indices),post_first_rng_sha256=tensor_bytes_hash(generator.get_state()),
        permutation_dtype=str(indices.dtype),permutation_count=indices.numel(),
        interpretation='Saved pre-rollout private shuffle RNG; source uses this generator only in device_minibatches after rollout')


def policy_ascent(actor,bank,obs,native,old_logprob,advantage,log_action_scale,args):
    logits=torch.func.functional_call(actor,bank,(obs,),strict=True)
    alpha,beta=(1+F.softplus(logits)).chunk(2,-1)
    distribution=Beta(alpha,beta,validate_args=False)
    ratio=((distribution.log_prob(native)-log_action_scale).sum(-1)-old_logprob).exp()
    if args.norm_adv:advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
    # Retain source maximum/clamp tie derivatives; a hard Boolean mask would
    # change the exact boundary law even if generic ratios usually avoid ties.
    return -torch.maximum(-advantage*ratio,-advantage*ratio.clamp(1-args.clip_coef,1+args.clip_coef_upper)).mean()


def full_covector_summary(ascent,negative_clipped_loss):
    if set(ascent)!=set(negative_clipped_loss):raise ValueError('Incomplete covector parameter coverage')
    uu=vv=uv=0.
    for name,left in ascent.items():
        right=negative_clipped_loss[name]
        if left.shape!=right.shape:raise ValueError('Covector shape mismatch')
        finite(left,right)
        uu+=float(left.double().square().sum());vv+=float(right.double().square().sum());uv+=float((left.double()*right.double()).sum())
    scale=uv/uu if uu>0 else None
    residual_squared=sum(float((negative_clipped_loss[name].double()-scale*left.double()).square().sum()) for name,left in ascent.items()) if scale is not None else None
    difference=sum(float((negative_clipped_loss[name].double()-left.double()).square().sum()) for name,left in ascent.items())**.5
    return dict(independent_policy_ascent_norm=uu**.5,negative_recorded_clipped_loss_norm=vv**.5,
        dot=uv,cosine=uv/(uu*vv)**.5 if uu>0 and vv>0 else None,
        unconstrained_least_squares_common_scale=scale,common_scale_positive=scale>0 if scale is not None else None,
        residual_after_common_scale_norm=residual_squared**.5 if residual_squared is not None else None,
        residual_relative_to_recorded_norm=(residual_squared/vv)**.5 if residual_squared is not None and vv>0 else None,
        raw_difference_norm=difference,
        scope='Independent compiled actor-only covector versus recorded negative clipped LOSS covector; LS scale is a comparison statistic, not an independently captured clipping factor or physical PG.')


def actor_routes(names):
    groups={}
    for name in names:
        label=name.rsplit('.',1)[0] if name.endswith(('.weight','.bias')) else name
        groups.setdefault(label,[]).append(name)
    return groups


def _audit_private(replay):
    args=replay.args;module=replay.module
    if module.__name__ not in MODULES or not args.compile or args.ent_coef!=0:
        raise ValueError('Require actual allowlisted compiled pure-policy actor gradient (entropy coefficient zero)')
    actor_ids={id(p) for p in replay.agent.actor.parameters()};critic_ids={id(p) for p in replay.agent.critic.parameters()}
    if actor_ids & critic_ids or actor_ids | critic_ids != {id(p) for p in replay.agent.parameters()}:
        raise ValueError('Require disjoint complete actor/critic trunks')
    agent=copy.deepcopy(replay.agent)
    for name,value in agent.state_dict().items():
        if not torch.equal(value,replay.snapshot['model'][name]):raise ValueError('Private old model differs from recorded source model')
    if tuple(agent.actor.named_buffers()):raise ValueError('Unspecified actor buffers')
    saved=replay.snapshot.get('rollout_start_rng',{}).get('shuffle')
    device=next(agent.parameters()).device
    if device.type!='cuda':raise ValueError('CUDA mandatory; execute through MLQ')
    indices,rng=exact_first_permutation(module,args,saved,device)
    batch=replay.snapshot['batch']
    fields=('obs','native','logprob','advantages','returns','values')
    selected={name:batch[name][indices].detach().clone(memory_format=torch.contiguous_format) for name in fields}
    finite(*selected.values())
    entry=replay.trace['trace'][0]
    if entry.get('epoch')!=0 or entry.get('update')!=0:raise ValueError('Trace lacks exact first update')
    recorded=entry['gradient_after_clip'];parameters=dict(agent.named_parameters())
    if set(recorded)!=set(parameters):raise ValueError('Missing recorded gradient parameter')
    def loss_model(obs,native,old_logprob,advantages,returns,old_values):
        return module.ppo_loss(agent,obs,native,old_logprob,advantages,returns,old_values,args)
    compiled=torch.compile(loss_model,mode=args.compile_mode,fullgraph=True,dynamic=False)
    torch.compiler.cudagraph_mark_step_begin()
    loss,metrics=compiled(*(selected[name] for name in fields))
    agent.zero_grad(set_to_none=True)
    loss.backward()
    saved_loss=loss.detach().clone();saved_metrics=metrics.detach().clone()
    if any(p.grad is None for p in parameters.values()):raise ValueError('Source loss did not produce complete actor/critic gradients')
    raw={name:p.grad.detach().clone() for name,p in parameters.items()}
    finite(loss,metrics,*raw.values())
    actor_preclip=torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()),args.max_grad_norm,foreach=True)
    critic_preclip=torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()),args.max_grad_norm,foreach=True)
    finite(actor_preclip,critic_preclip)
    clipped={name:p.grad.detach().clone() for name,p in parameters.items()}
    parity={}
    for name,value in clipped.items():
        target=recorded[name];finite(value,target)
        if target.shape!=value.shape or target.dtype!=value.dtype or target.device!=value.device or target.stride()!=value.stride() or not torch.equal(value,target):
            error=float((value.double()-target.double()).abs().max()) if target.shape==value.shape else None
            raise ValueError(f'First recorded clipped-gradient BITWISE replay failed: {name}; max_abs_error={error}')
        parity[name]=dict(bitwise_equal=True,norm=float(value.double().norm()))
    # Fresh actor-only compiled graph on exactly the SAME saved permutation.
    bank={name:p.detach().clone(memory_format=torch.contiguous_format) for name,p in agent.actor.named_parameters()}
    def independent(bank,obs,native,logprob,advantage):
        return policy_ascent(agent.actor,bank,obs,native,logprob,advantage,agent.log_action_scale,args)
    gradient=torch.compile(torch.func.grad(independent),mode=args.compile_mode,fullgraph=True,dynamic=False)
    torch.compiler.cudagraph_mark_step_begin()
    ascent={name:value.detach().clone() for name,value in gradient(bank,selected['obs'],selected['native'],selected['logprob'],selected['advantages']).items()}
    negative={name:-recorded['actor.'+name].detach().clone() for name in bank}
    finite(*ascent.values(),*negative.values())
    groups=actor_routes(tuple(bank))
    routes={name:full_covector_summary({key:ascent[key] for key in keys},{key:negative[key] for key in keys}) for name,keys in groups.items()}
    params={name:full_covector_summary({name:ascent[name]},{name:negative[name]}) for name in bank}
    scale=full_covector_summary(ascent,negative)['unconstrained_least_squares_common_scale']
    common_route_residuals={label:dict(parameter_names=keys,
        common_full_actor_scale_residual_norm=sum(float((negative[key].double()-scale*ascent[key].double()).square().sum()) for key in keys)**.5 if scale is not None else None)
        for label,keys in groups.items()}
    return dict(status='accepted_bitwise_source_first_gradient_reconstruction',rng=rng,first_gradient_parity=parity,
        source_reconstructed_preclip_norms=dict(actor=float(actor_preclip),critic=float(critic_preclip)),
        source_loss=float(saved_loss),source_metrics=saved_metrics.cpu().tolist(),
        full_actor_covector=full_covector_summary(ascent,negative),actor_routes=routes,actor_parameters=params,
        common_scale_route_residuals=common_route_residuals,
        raw_gradient_scope='Raw gradients and preclip norms were NOT saved; these are source-reconstructed values accepted through recorded clipped-gradient equality.',
        conventions=dict(entropy_coefficient=args.ent_coef,advantage_normalization=args.norm_adv,
            actor_critic_disjoint=True,max_grad_norm=args.max_grad_norm,compile_mode=args.compile_mode,
            full_saved_batch=True,same_exact_first_permutation=True,gradient_clip_count_per_trunk=1),
        limitations=['Independent actor-only graph can differ in compiler fusion/reduction roundoff; discrepancies are measured, not certified away by an arbitrary residual threshold.',
            'Common LS scale, even positive with a tiny residual, is not an independently recorded clipping multiplier.',
            'Source gradient is a clipped stored-surrogate loss covector; independent policy gradient uses the same saved GAE/actions, not physical PG or causal return credit.',
            'Shuffle RNG gives exact first permutation because the pinned source uses this private generator only after rollout; this is not a claim about unrecorded later gradient history.'])


def audit_first_gradient(replay):
    cpu=torch.get_rng_state().clone();device=next(replay.agent.parameters()).device
    if device.type!='cuda':raise ValueError('CUDA mandatory; execute through MLQ')
    cuda=torch.cuda.get_rng_state(device).clone()
    model={name:value.detach().clone() for name,value in replay.agent.state_dict().items()}
    saved_shuffle=replay.snapshot.get('rollout_start_rng',{}).get('shuffle')
    shuffle=saved_shuffle.detach().clone() if isinstance(saved_shuffle,torch.Tensor) else None
    # Compilation may use framework-global RNG internally; restore caller CPU
    # and this CUDA device on success AND exception. Policy draws use no global RNG.
    with torch.random.fork_rng(devices=[device.index],enabled=True):
        result=_audit_private(replay)
    if not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(cuda,torch.cuda.get_rng_state(device)):
        raise ValueError('Caller RNG changed despite private isolation')
    if any(not torch.equal(value,replay.agent.state_dict()[name]) for name,value in model.items()):raise ValueError('Caller model mutated')
    if shuffle is not None and not torch.equal(shuffle,saved_shuffle):raise ValueError('Saved shuffle RNG mutated')
    result['private_ownership']=dict(caller_model_unchanged=True,saved_shuffle_unchanged=True,caller_cpu_rng_unchanged=True,
        caller_cuda_rng_unchanged=True,cpu_rng_before_sha256=tensor_bytes_hash(cpu),cuda_rng_before_sha256=tensor_bytes_hash(cuda))
    return result


def analyze(snapshot_path,trace_path,**kwargs):
    sources={str(Path(__file__).resolve()):sha256(__file__),str(Path(__file__).with_name('sepclip_prescription_replay.py').resolve()):sha256(Path(__file__).with_name('sepclip_prescription_replay.py'))}
    replay=load_replay(snapshot_path,trace_path,**kwargs);result=audit_first_gradient(replay)
    for path,pin in ((snapshot_path,kwargs['snapshot_sha256']),(trace_path,kwargs['trace_sha256']),
                     (kwargs['manifest_path'],kwargs['manifest_sha256']),(kwargs['execution_path'],kwargs['execution_sha256'])):require_pin(path,pin)
    for path,pin in sources.items():require_pin(path,pin)
    validate_provenance(kwargs['module_name'],kwargs['manifest_path'],kwargs['manifest_sha256'],kwargs['execution_path'],kwargs['execution_sha256'],kwargs['exp_name'])
    result.update(source_sha256=sources,replay_validation=replay.validation,independently_recorded_logits=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','trace','manifest-path','execution-path','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('snapshot-sha256','trace-sha256','manifest-sha256','execution-sha256','exp-name'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--module-name',choices=tuple(MODULES),required=True)
    cli=vars(parser.parse_args());snapshot=cli.pop('snapshot');trace=cli.pop('trace');output=cli.pop('output')
    result=analyze(snapshot,trace,**cli);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

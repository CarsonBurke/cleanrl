"""Private actual-source warm optimizer replay using recorded clipped gradients.

CUDA/MLQ only. Reject unless each ambient/post-normalization delta and exact
endpoint meet declared contracts. Reconstructed moments are not recorded
ground truth, physical PG, hypothetical proposals or causal return evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sepclip_prescription_replay import MODULES, constraint_axes, load_replay, load_warm_optimizers, sha256, require_pin, validate_provenance


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError('Nonfinite optimizer forensic array')


def norm(value):
    finite(value)
    return float(value.double().norm())


def alignment(left,right):
    finite(left,right)
    a=norm(left);b=norm(right)
    return dict(left_norm=a,right_norm=b,dot=float((left.double()*right.double()).sum()),
        cosine=float((left.double()*right.double()).sum())/(a*b) if a>0 and b>0 else None)


def delta_contract(actual,recorded,basepoint,label,require_exact=True):
    """Strict FP32 same-source replay with explicit subtraction/storage budget.

    Per-element bound is2e-7*|basepoint|+2e-6*|recorded_delta|+1e-10.
    An independent1e-3 norm-relative guard rejects tiny motions hidden by the
    basepoint allowance. Null recorded delta requires exact zero replay.
    Actual recorded-step replay requires bitwise equality; the bounds apply
    ONLY to the independent internal-stage reconstruction, not saved trace.
    """
    if actual.shape!=recorded.shape or actual.shape!=basepoint.shape:
        raise ValueError('Replay delta shape mismatch')
    if any(value.dtype!=torch.float32 or value.device.type!='cuda' for value in (actual,recorded,basepoint)):
        raise ValueError('Require recorded FP32 CUDA delta/basepoint arrays')
    finite(actual,recorded,basepoint)
    error=(actual.double()-recorded.double()).abs()
    bound=2e-7*basepoint.double().abs()+2e-6*recorded.double().abs()+1e-10
    reference=norm(recorded);error_norm=float(error.norm())
    relative=error_norm/reference if reference>0 else None
    bitwise=bool(torch.equal(actual,recorded))
    if (require_exact and not bitwise) or bool((error>bound).any()) or (reference>0 and relative>1e-3) or (reference==0 and bool((actual!=0).any())):
        raise ValueError(f'Recorded update replay parity failed: {label}; max_abs_error={float(error.max())}, l2_error={error_norm}, relative_error={relative}, requires_bitwise={require_exact}')
    return dict(max_abs_error=float(error.max()),error_norm=error_norm,recorded_norm=reference,
        relative_error=relative,elementwise_budget='2e-7*abs(basepoint)+2e-6*abs(recorded_delta)+1e-10',
        relative_error_limit=1e-3,null_motion_requires_exact=True,bitwise_equal=bitwise,requires_bitwise_equal=require_exact)


def endpoint_contract(actual,recorded,label):
    finite(actual,recorded)
    if actual.shape!=recorded.shape or actual.dtype!=recorded.dtype:
        raise ValueError('Endpoint schema mismatch')
    if not torch.equal(actual,recorded):
        raise ValueError(f'Recorded exact endpoint replay failed: {label}; max_abs_error={float((actual.double()-recorded.double()).abs().max())}')
    return dict(max_abs_error=float((actual.double()-recorded.double()).abs().max()),
        bitwise_equal=True,requires_bitwise_equal=True)


def state_tree_equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and bool(torch.equal(left,right))
    if type(left)!=type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(state_tree_equal(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(state_tree_equal(a,b) for a,b in zip(left,right))
    return left==right


def require_factory_order(fresh,saved,label):
    if saved is None or len(saved.get('param_groups',[]))!=len(fresh['param_groups']):
        raise ValueError('Saved '+label+' group coverage mismatch')
    for expected,actual in zip(fresh['param_groups'],saved['param_groups']):
        if actual.get('params')!=expected['params']:
            raise ValueError('Saved '+label+' serialized parameter order differs from actual factory')


def private_warm(replay):
    """Optimizer loaders can alias same-device state; deep-copy BEFORE loading."""
    private=copy.copy(replay)
    private.agent=copy.deepcopy(replay.agent)
    private.snapshot=dict(replay.snapshot,
        optimizer=copy.deepcopy(replay.snapshot['optimizer']),
        matrix_optimizer=copy.deepcopy(replay.snapshot.get('matrix_optimizer')))
    private.validation=copy.deepcopy(replay.validation)
    for name,value in private.agent.state_dict().items():
        if not torch.equal(value,replay.snapshot['model'][name]):
            raise ValueError('Private old model differs from recorded snapshot')
    fresh_matrix,fresh_adam=private.module.make_optimizers(private.agent,private.args)
    require_factory_order(fresh_adam.state_dict(),private.snapshot['optimizer'],'Adam')
    if fresh_matrix is not None:require_factory_order(fresh_matrix.state_dict(),private.snapshot['matrix_optimizer'],'matrix')
    matrix,adam=load_warm_optimizers(private)
    return private,matrix,adam


def validate_clock(private,matrix,adam):
    args=private.args
    if args.num_minibatches!=1 or args.update_epochs!=10 or args.target_kl is not None or not args.compile:
        raise ValueError('Require compiled fixed ten-epoch full-batch forensic scope')
    horizon=private.module.episode_horizon(args.env_id) if args.staggered_starts and args.num_envs>1 else 0
    offset=private.snapshot['global_step']-horizon*args.num_envs
    if offset<=0 or offset%args.batch_size:
        raise ValueError('Snapshot step is not an exact source rollout boundary')
    iteration=offset//args.batch_size
    if not 1<=iteration<=args.num_iterations:
        raise ValueError('Snapshot iteration outside actual training horizon')
    expected_steps=(iteration-1)*args.update_epochs
    fraction=1-(iteration-1)/args.num_iterations if args.anneal_lr else 1.
    for group in adam.param_groups:
        if group['lr']!=fraction*args.learning_rate:
            raise ValueError('Saved current Adam LR differs from source annealing')
        for parameter in group['params']:
            state=adam.state.get(parameter,{})
            if expected_steps and (not state or float(state['step'])!=expected_steps):
                raise ValueError('Incomplete warm Adam history/counter')
            if not expected_steps and state and float(state['step'])!=0:
                raise ValueError('Unexpected cold Adam history')
    if matrix is not None:
        for group in matrix.param_groups:
            if group['steps']!=expected_steps or group['rollouts']!=iteration-1:
                raise ValueError('Saved matrix clock differs from source capture boundary')
            expected_lr=torch.tensor(fraction*args.normuon_lr,device=group['lr'].device,dtype=group['lr'].dtype)
            if not torch.equal(group['lr'],expected_lr):
                raise ValueError('Saved current matrix LR differs from source annealing')
            if expected_steps and group['rollout_pending']:
                raise ValueError('Captured prior rollout unexpectedly pending')
            if args.matrix_method=='normuon':
                for _,_,state,_ in matrix.stacks:
                    if bool((state['velocity'][1]!=0).any()):
                        raise ValueError('NorMuon unused slow rail is nonzero')
    return dict(iteration=iteration,expected_prior_steps=expected_steps,current_lr_fraction=fraction,
        snapshot_position='After current-iteration LR annealing and rollout; before begin_rollout(10)')


def factory_mapping(agent,matrix,adam):
    identities={id(value):name for name,value in agent.named_parameters()}
    groups=[];covered=[]
    if matrix is not None:
        for index,(_,parameters,state,scale) in enumerate(matrix.stacks):
            names=[identities[id(value)] for value in parameters]
            groups.append(dict(bank=index,parameter_names=names,shape=list(parameters[0].shape),
                aspect_scale=scale,retained_lane_axis='rows' if parameters[0].shape[0]>=parameters[0].shape[1] else 'columns',
                serialized_state_owner=names[0],velocity_shape=list(state['velocity'].shape),second_shape=list(state['second'].shape)))
            covered+=names
    auxiliary=[[identities[id(value)] for value in group['params']] for group in adam.param_groups]
    covered+=sum(auxiliary,[])
    if len(covered)!=len(set(covered)) or set(covered)!=set(identities.values()):
        raise ValueError('Actual factory parameter coverage/order is incomplete')
    serialized={}
    for label,optimizer in (('matrix',matrix),('Adam',adam)):
        if optimizer is None:continue
        identifiers={}
        for live,saved in zip(optimizer.param_groups,optimizer.state_dict()['param_groups']):
            for parameter,key in zip(live['params'],saved['params']):identifiers[str(key)]=identities[id(parameter)]
        serialized[label]=identifiers
    return dict(matrix_banks=groups,auxiliary_groups=auxiliary,serialized_parameter_ids=serialized)


def reconstruct_stages(module,old_weights,gradient,post_velocity,old_second,scalars,beta2,scale,method,precision):
    """Source-defined actual-method stages, not an alternative optimizer.

    Runs in a separately compiled graph. Its reproduced master subtraction
    must match the actual source step before accepting internal-stage telemetry.
    """
    lr,momentum,_,fast_weight,_=scalars.unbind()
    blend=(fast_weight*post_velocity[0]+(1-fast_weight)*post_velocity[1]) if method=='anvil2' else post_velocity[0]
    lookahead=gradient.lerp(blend,momentum)
    cascade=module.spectral_cascade(lookahead,method,precision)
    axis=-1 if gradient.shape[-2]>=gradient.shape[-1] else -2
    second=old_second.lerp(cascade.square().mean(axis,keepdim=True),1-beta2)
    adapted=cascade*second.clamp_min(1e-10).rsqrt()
    equalized=adapted*(cascade.norm(dim=(-2,-1),keepdim=True)/adapted.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-10))
    applied=equalized.bfloat16().float() if method=='anvil2' and precision=='bf16' else equalized
    proposal=-lr*scale*applied
    updated=old_weights.clone()
    updated.sub_(lr*scale*applied)
    return blend,lookahead,cascade,second,adapted,equalized,applied,proposal,updated


def adam_summary(agent,adam):
    names={id(value):name for name,value in agent.named_parameters()};rows={}
    for group in adam.param_groups:
        for parameter in group['params']:
            state=adam.state.get(parameter,{})
            if not state: continue
            rows[names[id(parameter)]]=dict(step=float(state['step']),exp_avg_norm=norm(state['exp_avg']),
                exp_avg_sq_min=float(state['exp_avg_sq'].min()),exp_avg_sq_max=float(state['exp_avg_sq'].max()),
                exp_avg_sq_norm=norm(state['exp_avg_sq']),
                gradient_moment=alignment(parameter.grad,state['exp_avg']) if parameter.grad is not None else None,
                source_betas=list(group['betas']),source_eps=group['eps'],source_lr=group['lr'])
    return rows


def audit_optimizer(replay):
    if not replay.module.__name__.endswith('_v14'):
        raise ValueError('Forensic stage reconstruction explicitly supports frozen v14 only')
    original_optimizer=copy.deepcopy(replay.snapshot['optimizer'])
    original_matrix=copy.deepcopy(replay.snapshot.get('matrix_optimizer'))
    private,matrix,adam=private_warm(replay)
    clock=validate_clock(private,matrix,adam)
    mapping=factory_mapping(private.agent,matrix,adam)
    entries=private.trace['trace']
    if len(entries)!=10: raise ValueError('Require complete ten-entry actual trace')
    parameters=dict(private.agent.named_parameters());rows=[]
    axes=constraint_axes(private.module.__name__,private.args,private.agent)
    normalize=torch.compile(private.agent.normalize_matrices,fullgraph=True,options={'triton.cudagraphs':False})
    stage_graphs=[]
    if matrix is not None:
        for group,_,_,scale in matrix.stacks:
            def stages(w,g,v,s,c,beta2=group['beta2'],scale=scale,method=group['method'],precision=group['precision']):
                return reconstruct_stages(private.module,w,g,v,s,c,beta2,scale,method,precision)
            stage_graphs.append(torch.compile(stages,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False}))
        matrix.begin_rollout(10)
    for index,entry in enumerate(entries):
        if entry['epoch']!=index or entry['update']!=index: raise ValueError('Incomplete trace ordering')
        old={name:value.detach().clone() for name,value in parameters.items()}
        for name,parameter in parameters.items():
            gradient=entry['gradient_after_clip'][name]
            finite(gradient)
            parameter.grad=gradient.detach().clone(memory_format=torch.contiguous_format)
        before_adam=adam_summary(private.agent,adam)
        before=[]
        if matrix is not None:
            for group,ps,state,_ in matrix.stacks:
                before.append(dict(velocity=state['velocity'].clone(),second=state['second'].clone(),
                    steps=group['steps'],rollout_pending=group['rollout_pending']))
            matrix.step()
        adam.step()
        ambient={name:value.detach().clone()-old[name] for name,value in parameters.items()}
        ambient_parity={name:delta_contract(ambient[name],entry['ambient_delta'][name],old[name],f'{index}.ambient.{name}') for name in parameters}
        normalize()
        actual={name:value.detach().clone()-old[name] for name,value in parameters.items()}
        actual_parity={name:delta_contract(actual[name],entry['retracted_delta'][name],old[name],f'{index}.actual.{name}') for name in parameters}
        banks=[]
        if matrix is not None:
            for bank,((group,ps,state,scale),previous,graph) in enumerate(zip(matrix.stacks,before,stage_graphs)):
                names=mapping['matrix_banks'][bank]['parameter_names']
                gradients=torch.stack([parameters[name].grad for name in names])
                with torch.no_grad():
                    stages=tuple(value.clone() for value in graph(torch.stack([old[name] for name in names]),gradients,state['velocity'].clone(),previous['second'].clone(),group['scalars'].clone()))
                finite(*stages)
                blend,lookahead,cascade,second,adapted,equalized,applied,proposal,reconstructed=stages
                if not torch.allclose(second,state['second'],rtol=2e-6,atol=0):
                    raise ValueError(f'Independent actual-method lane-stage replay mismatch; max_abs_error={float((second.double()-state["second"].double()).abs().max())}; rtol=2e-6, atol=0')
                # Compare rounded master subtraction, not ideal proposal vs a
                # stored subtraction whose relative rounding can dominate.
                stage_parity=delta_contract(reconstructed-torch.stack([old[name] for name in names]),
                    torch.stack([ambient[name] for name in names]),torch.stack([old[name] for name in names]),f'{index}.bank{bank}.stages',require_exact=False)
                members={}
                for member,name in enumerate(names):
                    singular=torch.linalg.svdvals(cascade[member].double())
                    members[name]=dict(clipped_gradient_norm=norm(gradients[member]),
                        fast_rail_before_norm=norm(previous['velocity'][0,member]),slow_rail_before_norm=norm(previous['velocity'][1,member]),
                        fast_rail_after_norm=norm(state['velocity'][0,member]),slow_rail_after_norm=norm(state['velocity'][1,member]),
                        gradient_blend=alignment(gradients[member],blend[member]),gradient_lookahead=alignment(gradients[member],lookahead[member]),
                        gradient_cascade=alignment(gradients[member],cascade[member]),cascade_adapted=alignment(cascade[member],adapted[member]),
                        cascade_equalized=alignment(cascade[member],equalized[member]),
                        cascade_singular_values=singular.cpu().tolist(),
                        lane_second_before_min=float(previous['second'][member].min()),lane_second_before_max=float(previous['second'][member].max()),
                        lane_second_after_min=float(state['second'][member].min()),lane_second_after_max=float(state['second'][member].max()),
                        lane_clamp_fraction=float((state['second'][member]<1e-10).double().mean()),
                        cascade_norm=norm(cascade[member]),adapted_norm=norm(adapted[member]),equalized_norm=norm(equalized[member]),
                        applied_direction_norm=norm(applied[member]),final_quantization_error_norm=norm(applied[member]-equalized[member]),
                        ideal_source_proposal_norm=norm(proposal[member]),actual_ambient_motion_norm=norm(ambient[name]),
                        master_subtraction_rounding_error_norm=norm((reconstructed[member]-old[name])-proposal[member]),
                        negative_clipped_gradient_dot_actual=-float((gradients[member].double()*ambient[name].double()).sum()))
                banks.append(dict(bank=bank,source_schedule_scalars=group['scalars'].cpu().tolist(),
                    prior_steps=previous['steps'],prior_rollout_pending=previous['rollout_pending'],after_steps=group['steps'],
                    method=group['method'],precision=group['precision'],beta2=group['beta2'],aspect_scale=scale,
                    reconstructed_stage_subtraction_parity=stage_parity,members=members))
        motion={name:dict(weight_before_norm=norm(old[name]),gradient_after_clip_norm=norm(parameters[name].grad),
            ambient_motion_norm=norm(ambient[name]),actual_motion_norm=norm(actual[name]),
            normalization_correction_norm=norm(actual[name]-ambient[name]),constraint_axis=axes.get(name),
            geometry='unit_axes' if name in axes else 'free_euclidean',
            negative_clipped_gradient_dot_ambient=-float((parameters[name].grad.double()*ambient[name].double()).sum()),
            negative_clipped_gradient_dot_actual=-float((parameters[name].grad.double()*actual[name].double()).sum())) for name in parameters}
        rows.append(dict(epoch=index,ambient_parity=ambient_parity,actual_parity=actual_parity,matrix_banks=banks,
            parameter_motion=motion,auxiliary_adam_before=before_adam,auxiliary_adam=adam_summary(private.agent,adam)))
    endpoint={name:endpoint_contract(value,private.trace['model_after_update'][name],name) for name,value in private.agent.state_dict().items()}
    # Everything remains private; fail if a loader/optimizer aliased captured state.
    for name,value in replay.agent.state_dict().items():
        if not torch.equal(value,replay.snapshot['model'][name]):raise ValueError('Caller model was mutated by private forensic replay')
    if not state_tree_equal(original_optimizer,replay.snapshot['optimizer']) or not state_tree_equal(original_matrix,replay.snapshot.get('matrix_optimizer')):
        raise ValueError('Caller optimizer snapshot was mutated by private replay')
    return dict(status='accepted_actual_update_reconstruction',clock=clock,factory_mapping=mapping,epochs=rows,
        endpoint_parity=endpoint,warm_validation=private.validation.get('warm_optimizer'),
        scope='Exact-source private optimizer replay of recorded clipped gradients; internal stages reconstructed, not independently captured. No gradient recomputation, physical rollouts, alternative optimizer proposals or efficacy claim.',
        limitations=['Warm-state schema checks alone are not continuation evidence: every recorded delta and final endpoint parity is required here.',
            'All-actor clipping context is shared across matrix body and Adam head/mix/gain; unclipped gradients and clipping factors were not saved.',
            'Independent compiled stage graph has its own lane/subtraction parity checks; no hidden stage is accepted solely because a different final path matched.',
            'Bank singular spectra and moment alignment refer to actual recorded gradients and source maps, not physical PG or global return-efficiency.'])


def analyze(snapshot_path,trace_path,**kwargs):
    sources={str(Path(__file__).resolve()):sha256(__file__),str(Path(__file__).with_name('sepclip_prescription_replay.py').resolve()):sha256(Path(__file__).with_name('sepclip_prescription_replay.py'))}
    replay=load_replay(snapshot_path,trace_path,**kwargs);result=audit_optimizer(replay)
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

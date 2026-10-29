"""CUDA-only local functional geometry audit; submit through mlq.

No training or environment scores. Each checkpoint is loaded into a private
model: one actual next-rollout optimizer proposal is compared with tangent
counterfactuals, then mapped through the actor Jacobian on fixed stored states.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl import ppo_continuous_action_sepclip_rollout_clock_anvil_v7 as trainer
from cleanrl.ppo_continuous_action_sepclip_fisher_head_v8 import beta_logit_fisher


def tangent(weight, vector, dim):
    """Orthogonal projection for unit rows (dim=1) or columns (dim=0)."""
    return vector - weight * (weight * vector).sum(dim, keepdim=True)


def compatible_clock_state(saved, clock, epochs):
    """Add v7 clock metadata to legacy v4 banks without altering their history."""
    result = copy.deepcopy(saved)
    for group in result['param_groups']:
        group.setdefault('slow_clock', clock)
        group.setdefault('rollout_pending', True)
        group.setdefault('rollout_decay', .98 ** epochs)
        group.setdefault('rollouts', 0)
    return result


def component_groups(parameter_names, hidden_names):
    groups = [(name, [name]) for name in sorted(hidden_names)]
    head = {'head.0.weight', 'head.0.bias', 'readout_gain'}
    groups.extend((name, [name]) for name in sorted(head & set(parameter_names)))
    rest = sorted(set(parameter_names) - set(hidden_names) - head)
    if rest:
        groups.append(('other_actor_parameters', rest))
    return groups


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


def validate_audit_protocol(args):
    if args.num_minibatches != 1 or args.normuon_network != 'actor' or args.matrix_method != 'anvil2':
        raise ValueError('audit requires actor-only full-batch Anvil checkpoint')
    if args.norm_adv or args.ent_coef != 0:
        raise ValueError('raw policy-gradient audit requires norm_adv=False and ent_coef=0')


def parallel_transport(old, new, vector, dim):
    """Exact product-sphere transport along shortest non-antipodal geodesics."""
    denominator = 1 + (old * new).sum(dim, keepdim=True)
    return vector - (vector * new).sum(dim, keepdim=True) / denominator * (old + new)


def retracted_delta(weight, proposal, dim):
    updated = weight + proposal
    return torch.nn.functional.normalize(updated, dim=dim) - weight


def fisher_gram(changes, fisher):
    """Gram entries include 1/2, so sum of all entries is joint quadratic KL."""
    paired = torch.stack([torch.stack(change.chunk(2, -1), -1) for change in changes])
    return .5 * torch.einsum('isak,sakl,jsal->ij', paired, fisher, paired) / paired.shape[1]


def distribution_motion(logits, change):
    alpha, beta = (1 + torch.nn.functional.softplus(logits)).chunk(2, -1)
    da, db = (logits.sigmoid() * change).chunk(2, -1)
    concentration = alpha + beta
    mean = alpha / concentration
    dmean = ((1 - mean) * da - mean * db) / concentration
    dlogc = (da + db) / concentration
    return dict(mean_rms=float(dmean.square().mean().sqrt()),
                log_concentration_rms=float(dlogc.square().mean().sqrt()),
                mean_log_concentration_change=float(dlogc.mean()))


def finite_distribution_motion(before, after):
    a, b = (1 + torch.nn.functional.softplus(before)).chunk(2, -1)
    na, nb = (1 + torch.nn.functional.softplus(after)).chunk(2, -1)
    mean_change = na/(na+nb) - a/(a+b)
    concentration_change = (na+nb).log() - (a+b).log()
    # Preserve actual FP32 policy shapes; promote only KL special-function algebra.
    old, new = torch.distributions.Beta(a.double(),b.double()), torch.distributions.Beta(na.double(),nb.double())
    return dict(mean_rms=float(mean_change.square().mean().sqrt()),
                log_concentration_rms=float(concentration_change.square().mean().sqrt()),
                mean_log_concentration_change=float(concentration_change.mean()),
                exact_kl=float(torch.distributions.kl_divergence(old,new).sum(-1).mean()),
                exact_kl_numeric_dtype='float64', policy_shape_dtype=str(a.dtype))


def policy_gain(logits, change, native, advantage):
    alpha, beta = (1 + torch.nn.functional.softplus(logits)).chunk(2, -1)
    sa, sb = logits.sigmoid().chunk(2, -1)
    common = torch.digamma(alpha + beta)
    score_a = sa * (native.log() - torch.digamma(alpha) + common)
    score_b = sb * (torch.log1p(-native) - torch.digamma(beta) + common)
    da, db = change.chunk(2, -1)
    return float((advantage * (score_a * da + score_b * db).sum(-1)).mean())


def mode_allocation(logits, change, native, advantage, fisher):
    """Split a local logit step into mean-only and concentration-only motion."""
    alpha,beta=(1+torch.nn.functional.softplus(logits)).chunk(2,-1)
    sa,sb=logits.sigmoid().chunk(2,-1)
    za,zb=change.chunk(2,-1)
    da,db=sa*za,sb*zb
    concentration=alpha+beta
    mean=alpha/concentration
    dc=da+db
    dmean=((1-mean)*da-mean*db)/concentration
    floor = torch.finfo(logits.dtype).tiny
    mean_part=torch.cat((concentration*dmean/sa.clamp_min(floor),-concentration*dmean/sb.clamp_min(floor)),-1)
    concentration_part=torch.cat((mean*dc/sa.clamp_min(floor),(1-mean)*dc/sb.clamp_min(floor)),-1)
    gram=fisher_gram([mean_part,concentration_part],fisher)
    return dict(mean_only_first_order_gain=policy_gain(logits,mean_part,native,advantage),
                concentration_only_first_order_gain=policy_gain(logits,concentration_part,native,advantage),
                mean_concentration_fisher_gram=gram.tolist(),
                additive_relative_error=float((mean_part+concentration_part-change).norm()/change.norm().clamp_min(1e-30)))


def cosine(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def audit(snapshot_path, subset_size, mode_count):
    snapshot = torch.load(snapshot_path, map_location='cuda', weights_only=False)
    args = SimpleNamespace(**snapshot['args'])
    validate_audit_protocol(args)
    args.compile = True
    args.slow_clock = getattr(args, 'slow_clock', 'epoch')
    batch = snapshot['batch']
    dimensions = batch['obs'].shape[-1]
    descriptor = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (dimensions,), dtype=np.float32),
        single_action_space=gym.spaces.Box(snapshot['model']['action_low'].cpu().numpy(),
                                          snapshot['model']['action_high'].cpu().numpy()))
    agent = trainer.Agent(descriptor).cuda()
    agent.load_state_dict(snapshot['model'])
    matrix, adam = trainer.make_optimizers(agent, args)
    matrix.load_state_dict(compatible_clock_state(snapshot['matrix_optimizer'], args.slow_clock, args.update_epochs))
    adam.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    original_model = copy.deepcopy(agent.state_dict())
    original_matrix = copy.deepcopy(matrix.state_dict())
    original_adam = copy.deepcopy(adam.state_dict())
    actor = agent.actor
    base = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    actor_names = {parameter: name for name, parameter in actor.named_parameters()}
    loss_fn = torch.compile(lambda *fields: trainer.ppo_loss(agent, *fields, args),
                            fullgraph=True, options={'triton.cudagraphs': False})
    agent.zero_grad(set_to_none=True)
    loss_fn(*(batch[name] for name in ('obs', 'native', 'logprob', 'advantages', 'returns', 'values')))[0].backward()
    raw_gradients = {name: parameter.grad.detach().clone() for name, parameter in actor.named_parameters()}
    torch.nn.utils.clip_grad_norm_(tuple(actor.parameters()), args.max_grad_norm, foreach=True)
    torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()), args.max_grad_norm, foreach=True)
    clipped = {parameter: parameter.grad.detach().clone() for parameter in actor.parameters()}
    matrix.begin_rollout(args.update_epochs)
    matrix.step()
    matrix_only = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    adam.step()
    ambient = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    agent.normalize_matrices()
    actual = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    count = min(subset_size, len(batch['obs']))
    generator = torch.Generator(device='cuda').manual_seed(91037)
    indices = torch.randperm(len(batch['obs']), generator=generator, device='cuda')[:count]
    obs, native, advantages = (batch[name][indices] for name in ('obs', 'native', 'advantages'))
    forward = torch.compile(lambda parameters: functional_call(actor, parameters, (obs,)),
                            fullgraph=True, options={'triton.cudagraphs': False})
    derivative = torch.compile(lambda parameters, direction: jvp(
        lambda p: functional_call(actor, p, (obs,)), (parameters,), (direction,))[1],
        fullgraph=True, options={'triton.cudagraphs': False})
    zeros = {name: torch.zeros_like(parameter) for name, parameter in base.items()}
    with torch.no_grad():
        logits = forward(base)
        fisher = beta_logit_fisher(logits)
        total_direction = {name: actual[name] - parameter for name, parameter in base.items()}
        total_jvp = derivative(base, total_direction)
        finite = forward(actual) - logits
        components, changes = [], []
        hidden_names = {actor_names[p] for _, parameters, _, _ in matrix.stacks for p in parameters}
        groups = component_groups(base, hidden_names)
        for label, names in groups:
            direction = dict(zeros)
            candidate = dict(base)
            for name in names:
                direction[name] = total_direction[name]
                candidate[name] = actual[name]
            change = derivative(base, direction)
            finite_change = forward(candidate) - logits
            changes.append(change)
            components.append(dict(component=label, parameter_delta_norm=float(torch.stack(
                [direction[name].square().sum() for name in names]).sum().sqrt()),
                train_subset_first_order_gain=policy_gain(logits, change, native, advantages),
                allocation=mode_allocation(logits,change,native,advantages,fisher),
                linear=distribution_motion(logits, change),
                finite=finite_distribution_motion(logits,logits+finite_change),
                finite_logit_linearization_relative_error=float((finite_change-change).norm()/change.norm().clamp_min(1e-30))))
        gram = fisher_gram(changes, fisher)
        banks = []
        for group, parameters, state, scale in matrix.stacks:
            lr, momentum, fast_beta, fast_weight, slow_beta = group['scalars'].unbind()
            for index, parameter in enumerate(parameters):
                name, weight, gradient = actor_names[parameter], base[actor_names[parameter]], clipped[parameter]
                dim = 0 if '.down.' in name else 1
                old_bank = original_matrix['state']
                # Original state banks are keyed by the first parameter's saved index.
                live_ids = matrix.param_groups[0]['params']
                saved_ids = original_matrix['param_groups'][0]['params']
                saved_key = saved_ids[next(i for i, p in enumerate(live_ids) if p is parameters[0])]
                prior = old_bank[saved_key]['velocity'][:, index]
                current = state['velocity'][:, index]
                blend = fast_weight * current[0] + (1-fast_weight) * current[1]
                lookahead = gradient.lerp(blend, momentum)
                proposed = weight - matrix_only[name]
                pgradient = tangent(weight, gradient, dim)
                projected_prior = torch.stack([tangent(weight, rail, dim) for rail in prior])
                projected_current = torch.stack((projected_prior[0].lerp(pgradient, 1-fast_beta),
                                                  projected_prior[1].lerp(pgradient, 1-slow_beta)))
                transported_blend = fast_weight*projected_current[0]+(1-fast_weight)*projected_current[1]
                projected_lookahead = pgradient.lerp(transported_blend, momentum)
                counter = trainer.spectral_cascade(projected_lookahead, group['method'], group['precision'])
                red_dim = -1 if weight.shape[0] >= weight.shape[1] else -2
                second = old_bank[saved_key]['second'][index].lerp(counter.square().mean(red_dim,keepdim=True),1-group['beta2'])
                adapted = counter * second.clamp_min(1e-10).rsqrt()
                counter = adapted * counter.norm()/adapted.norm().clamp_min(1e-10)
                if group['precision'] == 'bf16':
                    counter = counter.bfloat16().float()
                counter = tangent(weight, counter, dim) * lr * scale
                after_only = tangent(weight, proposed, dim)
                contrasts = []
                for label, proposal in [('actual_ambient_then_retract', -proposed),
                                         ('project_final_only', -after_only),
                                         ('project_gradient_and_stored_rails_before_polar', -counter)]:
                    delta = retracted_delta(weight, proposal, dim)
                    direction = dict(zeros); direction[name] = delta
                    change = derivative(base, direction)
                    candidate = dict(base); candidate[name] = weight + delta
                    contrasts.append(dict(case=label, delta_norm=float(delta.norm()),
                        raw_pg_descent_innerproduct=float(-(raw_gradients[name]*delta).sum()),
                        train_subset_first_order_gain=policy_gain(logits,change,native,advantages),
                        fisher_quadratic=float(fisher_gram([change],fisher)[0,0]),
                        motion=distribution_motion(logits,change),
                        finite=finite_distribution_motion(logits,forward(candidate))))
                u, singular, vh = torch.linalg.svd(lookahead, full_matrices=False)
                ranks = sorted(set(list(range(min(mode_count//2,len(singular))))+
                                   list(range(max(0,len(singular)-(mode_count+1)//2),len(singular)))))
                modes=[]
                for rank in ranks:
                    mode=u[:,rank,None]*vh[rank,None,:]
                    coefficient=(mode*proposed).sum()
                    delta=retracted_delta(weight,-coefficient*mode,dim)
                    direction=dict(zeros); direction[name]=delta
                    change=derivative(base,direction)
                    modes.append(dict(index=rank,singular_value=float(singular[rank]),
                        actual_proposal_component=float(coefficient),
                        clipped_gradient_component=float((mode*gradient).sum()),
                        train_subset_first_order_gain=policy_gain(logits,change,native,advantages),
                        fisher_quadratic=float(fisher_gram([change],fisher)[0,0]),
                        motion=distribution_motion(logits,change)))
                radial=lambda vector: float((vector-tangent(weight,vector,dim)).norm()/vector.norm().clamp_min(1e-30))
                banks.append(dict(parameter=name,axis=dim,gradient_radial_norm_fraction=radial(gradient),
                    rail_radial_norm_fractions=[radial(rail) for rail in prior],
                    lookahead_radial_norm_fraction=radial(lookahead),proposal_radial_norm_fraction=radial(proposed),
                    gradient_proposal_cosine=cosine(gradient,proposed),
                    tangent_gradient_proposal_cosine=cosine(pgradient,after_only),
                    pg_innerproduct_before_projection=float((gradient*proposed).sum()),
                    pg_innerproduct_after_projection=float((pgradient*after_only).sum()),
                    counterfactuals=contrasts,singular_modes=modes))
        result=dict(snapshot=str(snapshot_path),global_step=snapshot['global_step'],subset_count=count,
            actual_slow_clock=args.slow_clock,
            components=components,component_labels=[item['component'] for item in components],
            fisher_gram=gram.tolist(),joint_fisher_quadratic=float(gram.sum()),
            diagonal_fisher_quadratic=float(gram.diagonal().sum()),
            cross_fisher_quadratic=float(gram.sum()-gram.diagonal().sum()),
            jvp_additivity_relative_error=float((sum(changes)-total_jvp).norm()/total_jvp.norm().clamp_min(1e-30)),
            full_step_logit_linearization_relative_error=float((finite-total_jvp).norm()/total_jvp.norm().clamp_min(1e-30)),
            full_step_motion=distribution_motion(logits,total_jvp),
            full_step_allocation=mode_allocation(logits,total_jvp,native,advantages,fisher),
            full_step_finite=finite_distribution_motion(logits,logits+finite),banks=banks)
    # The model and optimizer are private, but restore them explicitly to verify
    # that multiple snapshots/cases cannot inherit a proposal or its gradients.
    agent.load_state_dict(original_model)
    matrix.load_state_dict(original_matrix)
    adam.load_state_dict(original_adam)
    agent.zero_grad(set_to_none=True)
    for name,value in agent.state_dict().items():
        torch.testing.assert_close(value,original_model[name],atol=0,rtol=0)
    assert_tree_equal(matrix.state_dict(), original_matrix)
    assert_tree_equal(adam.state_dict(), original_adam)
    assert all(parameter.grad is None for parameter in agent.parameters())
    result['model_restored_exactly']=True
    result['optimizer_and_initial_gradients_restored_exactly']=True
    return result


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--snapshot',action='append',required=True,type=Path)
    cli.add_argument('--out',required=True,type=Path)
    cli.add_argument('--subset-size',type=int,default=512)
    cli.add_argument('--singular-modes',type=int,default=6)
    cli.add_argument('--exp-name',default='sepclip_layer_transport_audit')
    args=cli.parse_args()
    if args.subset_size<=0 or args.singular_modes<=0:
        cli.error('subset size and singular modes must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run this audit through mlq')
    trainer.configure_runtime(matmul_precision='highest',allow_tf32=False)
    results=[]
    run_name=f'HalfCheetah-v4__{args.exp_name}__1__{int(time.time())}'
    with SummaryWriter(f'runs/{run_name}') as writer:
        for snapshot in args.snapshot:
            torch._dynamo.reset()
            record=audit(snapshot,args.subset_size,args.singular_modes)
            results.append(record)
            for name in ('joint_fisher_quadratic','cross_fisher_quadratic',
                         'full_step_logit_linearization_relative_error','jvp_additivity_relative_error'):
                writer.add_scalar(name,record[name],record['global_step'])
            writer.add_text(str(snapshot),json.dumps(record,allow_nan=False),record['global_step'])
    result=dict(snapshots=results,limitations=[
        'Stored training states/actions/GAE only: gains are not independent heldout or environment returns.',
        'One next-rollout full-batch proposal, not the entire ten-epoch optimizer trajectory; legacy v4 uses its epoch clock.',
        'Subset geometry is deterministic but noisy; no giant parameter Jacobians are constructed.',
        'Mean RMS is in native Beta units; physical action motion additionally multiplies each action range.',
        'Projected stored rails approximate transport to the current tangent; previous base points are absent.',
        'At one fixed base point projection is linear, so projecting lookahead equals projecting its gradient/rail constituents; multi-step transport benefits are not identified.',
        'Counterfactuals change optimizer geometry and amplitude; no improvement is claimed.',
        'Singular modes decompose lookahead; lane adaptation mixes them, and only selected diagonal components are shown.',
        'The Gram includes cross-layer terms locally; finite policy displacement need not equal its quadratic.',
        'Private model and optimizer state are restored; environment/RNG trainer continuation is not tested.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()

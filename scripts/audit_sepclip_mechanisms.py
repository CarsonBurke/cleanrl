"""Local geometry and held-out update diagnostics; queue all CUDA work via mlq.

Warm-clock replay switches existing Anvil rails. New noise covariance and
Fisher head histories are explicitly reset. These are not fresh-training scores.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import awr_update_bench as bench
import diagnose_sepclip_normuon as probe
from cleanrl import ppo_continuous_action_sepclip_anvil2_v4 as original
from cleanrl import ppo_continuous_action_sepclip_noise_polar_v6 as noise_module
from cleanrl import ppo_continuous_action_sepclip_rollout_clock_anvil_v7 as clock_module
from cleanrl import ppo_continuous_action_sepclip_fisher_head_v8 as fisher_module


@torch.no_grad()
def exact_policy_kl(before, after):
    old = torch.distributions.Beta(*before, validate_args=False)
    new = torch.distributions.Beta(*after, validate_args=False)
    return torch.distributions.kl_divergence(old, new).sum(-1).mean()


@torch.no_grad()
def effective_head(actor):
    layer, gain = actor.head[0], actor.readout_gain
    return torch.cat((gain[:, None] * layer.weight, (gain * layer.bias)[:, None]), -1)


@torch.no_grad()
def logit_fisher_quadratic(before_logits, after_logits):
    change_a, change_b = (after_logits-before_logits).chunk(2, -1)
    change = torch.stack((change_a, change_b), -1)
    fisher = fisher_module.beta_logit_fisher(before_logits)
    return .5 * (change * (fisher @ change[..., None]).squeeze(-1)).sum((-1, -2)).mean()


@torch.no_grad()
def local_policy_geometry(before_logits, after_logits, batch, log_action_scale):
    """Actual logit displacement, local policy score and exact incremental KL.

    The first-order gains concern this heldout batch, not environment return.
    Sample Fisher is exact at the old logits; finite displacement is nonlinear.
    """
    a, b = (1+torch.nn.functional.softplus(before_logits)).chunk(2, -1)
    old = torch.distributions.Beta(a, b, validate_args=False)
    common = torch.digamma(a+b)
    sa, sb = before_logits.sigmoid().chunk(2, -1)
    score_a = sa*(batch['native'].log()-torch.digamma(a)+common)
    score_b = sb*(torch.log1p(-batch['native'])-torch.digamma(b)+common)
    da, db = (after_logits-before_logits).chunk(2, -1)
    local_logratio = (score_a*da+score_b*db).sum(-1)
    physical_logprob = (old.log_prob(batch['native'])-log_action_scale).sum(-1)
    ratio = (physical_logprob-batch['logprob']).exp()
    gae = batch['advantages']*ratio*local_logratio
    clipped_active = ((batch['advantages']>0)&(ratio>1.2))|((batch['advantages']<0)&(ratio<.8))
    next_a, next_b = (1+torch.nn.functional.softplus(after_logits)).chunk(2, -1)
    exact_kl = exact_policy_kl((a,b), (next_a,next_b))
    return dict(first_order_gae_gain=float(gae.mean()),
        first_order_clipped_gae_gain=float(torch.where(clipped_active,0.,gae).mean()),
        first_order_lambda1_gain=float((batch['advantages_mc']*ratio*local_logratio).mean()),
        sample_fisher_quadratic=float(logit_fisher_quadratic(before_logits,after_logits)),
        exact_incremental_kl=float(exact_kl))


def copy_adam_subset(agent, old_optimizer, saved, new_optimizer):
    """Map original Adam state by actual parameter objects, never position after filtering."""
    index_to_parameter = dict(zip(saved['param_groups'][0]['params'], old_optimizer.param_groups[0]['params']))
    available = {parameter: saved['state'][index] for index, parameter in index_to_parameter.items()
                 if index in saved['state']}
    for group in new_optimizer.param_groups:
        group['lr'] = old_optimizer.param_groups[0]['lr']
        for parameter in group['params']:
            if parameter in available:
                new_optimizer.state[parameter] = copy.deepcopy(available[parameter])


def run(path, out, writer):
    snapshot = torch.load(path, map_location='cuda', weights_only=False)
    args = SimpleNamespace(**snapshot['args'])
    args.num_iterations = (args.total_timesteps - 16000) // (args.num_steps * args.num_envs)
    args.slow_clock = 'rollout'
    args.noise_multiplier, args.lane_equalize = 1., False
    args.actor_matrix_optimizer, args.head_damping, args.head_target_kl = 'anvil2', .01, .001
    envs = original.make_training_env(args, 'sepclip_mechanisms_audit')
    try:
        agent = fisher_module.Agent(envs).cuda()
    finally:
        envs.close()
    agent.load_state_dict(snapshot['model'])
    base = copy.deepcopy(agent.state_dict())
    train = snapshot['batch']
    with torch.no_grad():
        train['alpha'], train['beta'], _ = agent.get_policy_and_value(train['obs'])
        train['advantages_mc'] = train['advantages']
        chunks, returns = bench.collect(fisher_module, agent, snapshot, 3, 'cuda', 1)
        for chunk in chunks:
            chunk['values'] = agent.get_value(chunk['obs']).flatten()
            chunk['returns'] = chunk['values'] + chunk['advantages']
    hold = bench.concatenate(chunks)
    loss = torch.compile(lambda *fields: original.ppo_loss(agent, *fields, args), fullgraph=True,
                         options={'triton.cudagraphs': False})
    train_gradients = probe.gradients(agent, train, loss)
    held_gradients = probe.gradients(agent, hold, loss)
    fields = tuple(train[name] for name in ('obs', 'native', 'logprob', 'advantages', 'returns', 'values'))
    folds = noise_module.environment_folds(args.num_steps, args.num_envs,
        torch.Generator(device='cuda').manual_seed(17042), 'cuda')
    old_matrix, old_adam = original.make_optimizers(agent, args)
    old_adam.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    actor_parameters = tuple(agent.actor.parameters())
    critic_parameters = tuple(agent.critic.parameters())
    matrix_parameters = {parameter for group in old_matrix.param_groups for parameter in group['params']}
    names = {parameter: name for name, parameter in agent.named_parameters()}
    gradient_rows = []
    _, noises = noise_module.fold_gradients(loss, fields, folds, tuple(agent.parameters()), {id(p) for p in matrix_parameters})
    for parameter in sorted(matrix_parameters, key=names.get):
        g, n = train_gradients[names[parameter]], noises[parameter]
        filtered = noise_module.soft_polar(g, noise_module.noise_covariance(n))
        full_polar = noise_module.soft_polar(g, torch.zeros_like(noise_module.noise_covariance(n)), 0.)
        x = g.T if g.shape[0] > g.shape[1] else g
        eigenvalues = torch.linalg.eigvalsh(x @ x.T)
        floor = eigenvalues.abs().max() * min(g.shape) * torch.finfo(torch.float32).eps
        held = held_gradients[names[parameter]]
        cosine = lambda a, b: float((a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-20))
        gradient_rows.append(dict(parameter=names[parameter], noise_to_signal=float(n.norm()/g.norm().clamp_min(1e-20)),
            raw_heldout_cosine=cosine(g, held), full_polar_heldout_cosine=cosine(full_polar, held),
            filtered_heldout_cosine=cosine(filtered, held), retained_norm=float(filtered.norm()/full_polar.norm().clamp_min(1e-20)),
            floored_zero_noise_modes=int((eigenvalues < floor).sum()), rank=min(g.shape)))
    # Repartition one fixed batch: this measures estimator sensitivity, not
    # independent rollout noise or cross-seed uncertainty.
    partition_rows = []
    generator = torch.Generator(device='cuda').manual_seed(17319)
    for partition in range(8):
        repeated_folds = noise_module.environment_folds(args.num_steps, args.num_envs, generator, 'cuda')
        _, repeated_noises = noise_module.fold_gradients(loss, fields, repeated_folds,
            tuple(agent.parameters()), {id(p) for p in matrix_parameters})
        for parameter in sorted(matrix_parameters, key=names.get):
            gradient, held = train_gradients[names[parameter]], held_gradients[names[parameter]]
            covariance = noise_module.noise_covariance(repeated_noises[parameter])
            filtered = noise_module.soft_polar(gradient, covariance)
            isotropic = torch.eye(covariance.shape[-1], device='cuda') * covariance.diagonal().mean()
            scalar_filtered = noise_module.soft_polar(gradient, isotropic)
            axis = 0 if '.down.' in names[parameter] else 1
            tangent = filtered - parameter * (parameter * filtered).sum(axis, keepdim=True)
            partition_rows.append(dict(partition=partition, parameter=names[parameter],
                noise_to_signal=float(repeated_noises[parameter].norm()/gradient.norm().clamp_min(1e-20)),
                tangent_heldout_cosine=float((tangent*held).sum()/(tangent.norm()*held.norm()).clamp_min(1e-20)),
                filtered_heldout_cosine=float((filtered*held).sum()/(filtered.norm()*held.norm()).clamp_min(1e-20)),
                isotropic_heldout_cosine=float((scalar_filtered*held).sum()/(scalar_filtered.norm()*held.norm()).clamp_min(1e-20)),
                direction_norm=float(filtered.norm()), isotropic_direction_norm=float(scalar_filtered.norm())))
    configs = [('actual_anvil_warm', 1.), ('rollout_clock_switched_warm', 1.),
               ('noise_zero_cold', 2.), ('noise_full_cold', 2.), ('noise_low_cold', 2.),
               ('noise_full_cold', 4.), ('noise_equalized_cold', 2.),
               ('fisher_anvil_warm', 1.), ('fisher_anvil_small_warm', 1.), ('fisher_adam_trunk_reset', 1.)]
    current_lr = float(snapshot['matrix_optimizer']['param_groups'][0]['lr'])
    cases = []
    for kind, multiplier in configs:
        agent.load_state_dict(base)
        args.noise_multiplier, args.lane_equalize = (0. if kind == 'noise_zero_cold' else .3 if kind == 'noise_low_cold' else 1.), kind == 'noise_equalized_cold'
        args.actor_matrix_optimizer = 'adam' if kind == 'fisher_adam_trunk_reset' else 'anvil2'
        head = None
        if kind.startswith('noise'):
            args.normuon_momentum = .85
            args.normuon_lr = current_lr * multiplier
            opt, adam = noise_module.make_optimizers(agent, args)
            adam.load_state_dict(copy.deepcopy(snapshot['optimizer']))
            opt.begin_rollout()
        elif kind.startswith('rollout'):
            opt, adam = clock_module.make_optimizers(agent, args)
            saved = copy.deepcopy(snapshot['matrix_optimizer'])
            for group in saved['param_groups']:
                group.update(slow_clock='rollout', rollout_pending=True, rollout_decay=.98**10, rollouts=0)
            opt.load_state_dict(saved)
            adam.load_state_dict(copy.deepcopy(snapshot['optimizer']))
            opt.begin_rollout(args.update_epochs)
        elif kind.startswith('fisher'):
            opt, adam = fisher_module.make_optimizers(agent, args)
            if opt is not None:
                opt.load_state_dict(copy.deepcopy(snapshot['matrix_optimizer']))
            copy_adam_subset(agent, old_adam, snapshot['optimizer'], adam)
            radius = (.0003 if kind == 'fisher_anvil_small_warm' else .001) * (current_lr/snapshot['args']['normuon_lr'])**2
            with torch.no_grad():
                layer, gain = agent.actor.head[0], agent.actor.readout_gain
                fisher_module.factorize_effective_head(layer.weight, layer.bias, gain,
                    gain[:, None] * layer.weight, gain * layer.bias)
            head = fisher_module.FisherHead(agent.actor, args.head_damping, radius)
            with torch.no_grad():
                features = agent.actor.features(train['obs'])
                head.refresh(features, agent.actor.readout_gain * agent.actor.head(features))
        else:
            opt, adam = original.make_optimizers(agent, args)
            opt.load_state_dict(copy.deepcopy(snapshot['matrix_optimizer']))
            adam.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        epochs = []
        previous_head_delta = previous_head_gradient = None
        initial_noise = None
        for epoch in range(args.update_epochs):
            with torch.no_grad():
                pre_step_logits = agent.actor(hold['obs'])
            if kind.startswith('noise') and epoch == 0:
                _, frozen_noise = noise_module.fold_gradients(loss, fields, folds, tuple(agent.parameters()),
                                                               {id(p) for p in matrix_parameters})
            else:
                probe.gradients(agent, train, loss)
            noise_drift = None
            if kind == 'actual_anvil_warm' or (kind == 'noise_full_cold' and multiplier == 2.):
                # Diagnose covariance drift without changing the candidate's
                # actual gradient or frozen-noise update rule.
                saved_gradients = {p: p.grad.clone() for p in agent.parameters()}
                _, current_noise = noise_module.fold_gradients(loss, fields, folds,
                    tuple(agent.parameters()), {id(p) for p in matrix_parameters})
                for parameter, gradient in saved_gradients.items():
                    parameter.grad = gradient
                if initial_noise is None:
                    initial_noise = {p: n.clone() for p, n in current_noise.items()}
                noise_drift = []
                for parameter in sorted(matrix_parameters, key=names.get):
                    current, initial = current_noise[parameter], initial_noise[parameter]
                    current_covariance, initial_covariance = map(noise_module.noise_covariance, (current, initial))
                    noise_drift.append(dict(parameter=names[parameter],
                        initial_noise_cosine=float((current*initial).sum()/(current.norm()*initial.norm()).clamp_min(1e-20)),
                        covariance_relative_drift=float((current_covariance-initial_covariance).norm()/initial_covariance.norm().clamp_min(1e-20)),
                        noise_to_gradient=float(current.norm()/parameter.grad.norm().clamp_min(1e-20))))
            actor_norm = torch.nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
            torch.nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
            if kind.startswith('noise'):
                factor = (args.max_grad_norm / (actor_norm+1e-6)).clamp(max=1.)
                opt.step({parameter: noise*factor for parameter, noise in frozen_noise.items()})
            elif opt is not None:
                opt.step()
            adam.step()
            head_record = None
            if head is not None:
                with torch.no_grad():
                    before_policy = agent.get_policy_and_value(hold['obs'])[:2]
                    before_logits = agent.actor(hold['obs'])
                    train_before_policy = agent.get_policy_and_value(train['obs'])[:2]
                    train_features = agent.actor.features(train['obs'])
                    train_before_logits = agent.actor.readout_gain * agent.actor.head(train_features)
                    current_a, current_b, _, _ = fisher_module.head_covariances(train_features, train_before_logits, args.head_damping)
                    before_effective = effective_head(agent.actor)
                    layer, gain = agent.actor.head[0], agent.actor.readout_gain
                    gradient = torch.cat((layer.weight.grad/gain[:, None],
                        (layer.bias.grad/gain)[:, None]), -1).clone()
                    pre_head_train = probe.evaluate(agent, train)
                    pre_head_train.pop('mc_gain')
                    pre_head_hold = probe.evaluate(agent, hold)
                head.step()
                with torch.no_grad():
                    after_policy = agent.get_policy_and_value(hold['obs'])[:2]
                    exact_local_quadratic = logit_fisher_quadratic(before_logits, agent.actor(hold['obs']))
                    train_after_policy = agent.get_policy_and_value(train['obs'])[:2]
                    train_local_quadratic = logit_fisher_quadratic(train_before_logits, agent.actor(train['obs']))
                    delta = effective_head(agent.actor) - before_effective
                    delta_a, delta_b = delta.chunk(2, 0)
                    paired_delta = torch.stack((delta_a, delta_b), 1)
                    current_kfac_quadratic = .5 * (paired_delta * (current_b @ paired_delta @ current_a)).sum()
                    actual_head_kl = exact_policy_kl(before_policy, after_policy)
                    head_record = dict(predicted_kl=float(head.predicted_kl), exact_head_only_heldout_kl=float(actual_head_kl),
                        exact_sample_fisher_quadratic=float(exact_local_quadratic),
                        exact_to_local_quadratic_ratio=float(actual_head_kl/exact_local_quadratic.clamp_min(1e-20)),
                        exact_head_only_train_kl=float(exact_policy_kl(train_before_policy, train_after_policy)),
                        exact_sample_train_fisher_quadratic=float(train_local_quadratic),
                        current_train_kfac_quadratic=float(current_kfac_quadratic),
                        exact_to_predicted_ratio=float(actual_head_kl/head.predicted_kl.clamp_min(1e-20)),
                        effective_gradient_norm=float(gradient.norm()), effective_delta_norm=float(delta.norm()),
                        pre_head_train=pre_head_train, pre_head_hold=pre_head_hold,
                        post_head_before_normalization_hold=probe.evaluate(agent, hold))
                    if previous_head_delta is not None:
                        head_record['delta_previous_cosine'] = float((delta*previous_head_delta).sum()/
                            (delta.norm()*previous_head_delta.norm()).clamp_min(1e-20))
                        head_record['gradient_previous_cosine'] = float((gradient*previous_head_gradient).sum()/
                            (gradient.norm()*previous_head_gradient.norm()).clamp_min(1e-20))
                    previous_head_delta, previous_head_gradient = delta, gradient
            agent.normalize_matrices()
            train_result = probe.evaluate(agent, train)
            train_result.pop('mc_gain')
            with torch.no_grad():
                step_geometry = local_policy_geometry(pre_step_logits, agent.actor(hold['obs']), hold,
                                                      agent.log_action_scale)
            epochs.append(dict(epoch=epoch+1, train=train_result, hold=probe.evaluate(agent, hold),
                               local_policy_geometry=step_geometry))
            if head_record is not None:
                epochs[-1]['head'] = head_record
            if noise_drift is not None:
                epochs[-1]['noise_drift'] = noise_drift
        label = f'{kind}_rate{multiplier}'
        cases.append(dict(label=label, effective_matrix_lr=float(opt.param_groups[0]['lr']) if opt is not None else None,
                          effective_auxiliary_lr=float(adam.param_groups[0]['lr']), epochs=epochs))
        for tag in ('exact_kl', 'surrogate_gain', 'mc_gain'):
            for record in epochs:
                writer.add_scalar(f'{label}/{tag}', record['hold'][tag], record['epoch'])
        print(json.dumps(dict(label=label, final=epochs[-1])), flush=True)
    result = dict(snapshot=str(path), gradients=gradient_rows, covariance_partitions=partition_rows,
                  cases=cases, rollout_returns=returns,
                  limitations=['Frozen local replay, not fresh-training scores.',
                               'Noise states reset; rollout clock switches actual Anvil rails.',
                               'Natural-head statistics reset; Adam trunk state reset when absent in Anvil checkpoint.',
                               'Three disjoint heldout chunks are not independent training seeds.',
                               'Normalizer statistics reset at chunk boundaries and update within each chunk.',
                               'Heldout lambda-1 gains retain critic bootstrap bias and rollout correlation.',
                               'Covariance model assumes fixed per-rollout linearized gradient noise.',
                               'Head radius is Kronecker-predicted head KL, not total actor KL.',
                               'Zero-noise polar uses a FP32 Gram floor and attenuates numerical tail modes.',
                               'Repeated partitions share one fixed batch; they are estimator sensitivity checks.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    cli = parser.parse_args()
    original.configure_runtime(matmul_precision='highest', allow_tf32=False)
    torch._dynamo.config.recompile_limit = 128
    torch.manual_seed(1)
    with SummaryWriter(f'runs/HalfCheetah-v4__sepclip_mechanisms_diagnostic__1__{int(time.time())}') as writer:
        run(cli.snapshot, cli.out, writer)


if __name__ == '__main__':
    main()

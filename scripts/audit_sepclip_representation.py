"""CUDA representation audit; submit through mlq, never launch as bare ML work.

Visited observations and deliberately synthetic sign/radial perturbations are
reported separately. These diagnostics cannot establish a benchmark ceiling.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.host_actor import situ_glu
from cleanrl.shared.runtime import configure_runtime


def summary(tensor):
    original = tensor.detach().double().flatten()
    x = original[torch.isfinite(original)]
    if x.numel() == 0:
        return dict(mean=None, rms=None, quantiles=None, finite_count=0, total_count=original.numel())
    q = torch.quantile(x, torch.tensor([0., .05, .25, .5, .75, .95, 1.], device=x.device, dtype=x.dtype))
    return dict(mean=float(x.mean()), rms=float(x.square().mean().sqrt()),
                finite_count=x.numel(), total_count=original.numel(), quantiles=dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'), q.cpu().tolist())))


def spectra(matrix):
    """Singular spectra via a smaller Gram; no decomposition of model weights."""
    x = matrix.double()
    gram = x @ x.mT if x.shape[-2] <= x.shape[-1] else x.mT @ x
    energy = torch.linalg.eigvalsh(gram).clamp_min(0).flip(-1)
    raw_total = energy.sum(-1)
    total = raw_total.clamp_min(1e-30)
    fractions = energy / total[..., None]
    effective = torch.where(raw_total > 0,
        torch.exp(-(fractions * fractions.clamp_min(1e-30).log()).sum(-1)), torch.zeros_like(raw_total))
    # This is a reported diagnostic threshold, never used to alter the model.
    rank = (energy > energy[..., :1] * 1e-8).sum(-1)
    return dict(singular_values_mean=energy.sqrt().mean(tuple(range(energy.ndim-1))).cpu().tolist()
                if energy.ndim > 1 else energy.sqrt().cpu().tolist(),
                effective_rank=summary(effective), numerical_rank_relative_1e8=summary(rank),
                top1_energy=summary(fractions[..., 0]), top5_energy=summary(fractions[..., :5].sum(-1)))


def stages(actor, obs):
    raw = actor.first(obs)
    first = F.normalize(raw, dim=-1)
    branch_raw = actor.second(8 * first)
    branch = F.normalize(branch_raw, dim=-1)
    final = F.normalize(first + branch, dim=-1)
    return torch.stack((raw, first, branch_raw, branch, final), dim=-2)


def beta_quantities(agent, obs):
    alpha, beta, _ = agent.get_policy_and_value(obs)
    c = alpha + beta
    mean = alpha / c
    variance = alpha * beta / (c.square() * (c + 1))
    excess = (alpha - 1) + (beta - 1)
    mode = torch.where(excess > 0, (alpha - 1) / excess.clamp_min(1e-30),
                       torch.full_like(c, float('nan')))
    return alpha, beta, c, 2 * mean - 1, 4 * variance, 2 * mode - 1


def head_report(agent, obs, seed):
    with torch.no_grad():
        logits = agent.actor(obs)
        alpha, beta, concentration, mean, variance, mode = beta_quantities(agent, obs)
        gain = agent.actor.readout_gain
        weight = agent.actor.head[0].weight
        action_dim = agent.action_dim
        effective_weight = gain[:, None] * weight
        rows_a, rows_b = effective_weight[:action_dim], effective_weight[action_dim:]
        row_cosine = F.cosine_similarity(rows_a, rows_b, dim=-1)
        row_cosine = torch.where((rows_a.norm(dim=-1)>0) & (rows_b.norm(dim=-1)>0),
                                 row_cosine, torch.full_like(row_cosine, float('nan')))
        # Sampling uses its own host RNG and matches the independent Beta law.
        rng = np.random.default_rng(seed)
        sampled = 2 * rng.beta(alpha.cpu().numpy(), beta.cpu().numpy()) - 1
        torques = dict(sampled=torch.as_tensor(sampled, device='cuda'), mean=mean, mode=mode)
        return dict(logits=summary(logits), gain=summary(gain), alpha=summary(alpha), beta=summary(beta),
                    concentration=summary(concentration), physical_mean=summary(mean),
                    physical_variance=summary(variance), physical_mode=summary(mode),
                    reachable_abs_mean_upper_bound=summary(1 - 2 / concentration),
                    mean_bound_slack=summary((1 - 2 / concentration) - mean.abs()),
                    alpha_beta_head_row_cosine=summary(row_cosine),
                    effective_head_spectra=spectra(effective_weight),
                    mode_undefined_count=int((~torch.isfinite(mode)).sum()),
                    torque_proportions={name: {
                        str(cut): float((actions[torch.isfinite(actions)].abs() >= cut).float().mean())
                        if torch.isfinite(actions).any() else None
                        for cut in (.8, .9, .95, .98)} for name, actions in torques.items()},
                    torque_finite_counts={name: int(torch.isfinite(actions).sum())
                                          for name, actions in torques.items()})


def make_agent(module, snapshot):
    state = snapshot['model']
    obs_dim = state['actor.first.0.gate.weight'].shape[-1]
    low, high = state['action_low'].detach().cpu().numpy(), state['action_high'].detach().cpu().numpy()
    if not (np.allclose(low, -1) and np.allclose(high, 1)):
        raise ValueError('This physical Beta audit requires action bounds [-1, 1]')
    envs = SimpleNamespace(single_action_space=gym.spaces.Box(low, high, dtype=np.float32),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32))
    return module.Agent(envs).cuda().eval().requires_grad_(False)


def initialized_gate_calibration(agent, obs):
    with torch.no_grad():
        first = F.normalize(agent.actor.first(obs), dim=-1)
        result = {}
        for name, branch, inputs in [('first', agent.actor.first[0], obs),
                                     ('second', agent.actor.second[0], 8*first)]:
            gate, up = branch.gate(inputs), branch.up(inputs)
            result[name] = dict(
                input_norm=summary(inputs.norm(dim=-1)),
                gate_preactivation=summary(gate), up_preactivation=summary(up),
                gate_channel_variance=summary(gate.var(0, correction=0)),
                up_channel_variance=summary(up.var(0, correction=0)),
                gate_row_norm=summary(branch.gate.weight.norm(dim=1)),
                up_row_norm=summary(branch.up.weight.norm(dim=1)),
                down_column_norm=summary(branch.down.weight.norm(dim=0)),
                gate_sigmoid=summary(gate.sigmoid()),
                gate_cap_derivative=summary(1-torch.tanh(gate/4).square()),
                up_cap_derivative=summary(1-torch.tanh(up/25).square()),
                design_point_variance=2.0,
                qualification='Observed on the selected trained-rollout inputs, not assumed isotropic initialization data.')
        return result


def initialized_report(module, snapshot, obs, seed):
    # Fork global construction RNGs; all subset/sampling RNGs are independent.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        agent = make_agent(module, snapshot)
    with torch.no_grad():
        unprojected = copy.deepcopy(agent.state_dict())
        before = head_report(agent, obs, seed)
        before["trunk_gate_calibration"] = initialized_gate_calibration(agent, obs)
        head = agent.actor.head[0].weight
        head.div_(head.norm(dim=1, keepdim=True))
        head_only = head_report(agent, obs, seed)
        head_only["trunk_gate_calibration"] = initialized_gate_calibration(agent, obs)
        agent.load_state_dict(unprojected)
        agent.normalize_matrices()
        all_projected = head_report(agent, obs, seed)
        all_projected["trunk_gate_calibration"] = initialized_gate_calibration(agent, obs)
    return dict(input_label='actual snapshot observations applied to freshly initialized networks',
                unprojected=before, head_only_projected=head_only, all_matrices_projected=all_projected,
                exact_initial_head_bound='Each row has unit norm, gain=1, bias=0 and final hidden norm=8; therefore |logit| <= 8.')


def visited_report(agent, obs, jacobian_obs):
    names = ('first_raw', 'first_unit', 'branch_raw', 'branch_unit', 'final_unit')
    with torch.no_grad():
        features = stages(agent.actor, obs)
        sign_features = stages(agent.actor, -obs)
        feature_stats = {}
        for i, name in enumerate(names):
            values = features[:, i]
            centered = values - values.mean(0)
            feature_stats[name] = dict(norm=summary(values.norm(dim=-1)),
                                       centered_feature_spectra=spectra(centered),
                                       mean_direction_energy_fraction=float(values.mean(0).square().sum()
                                                                            / values.square().sum(-1).mean().clamp_min(1e-30)))
        gates = {}
        for name, branch, inputs in [('first', agent.actor.first[0], obs),
                                     ('second', agent.actor.second[0], 8 * features[:, 1])]:
            g, u = branch.gate(inputs), branch.up(inputs)
            even = .5 * (4 * torch.tanh(g/4)) * (25 * torch.tanh(u/25))
            odd = even * torch.tanh(g/2)
            actual = situ_glu(g, u)
            gates[name] = dict(gate=summary(g), up=summary(u), gate_sigmoid=summary(g.sigmoid()),
                               gate_cap_tanh_derivative=summary(1-torch.tanh(g/4).square()),
                               up_cap_tanh_derivative=summary(1-torch.tanh(u/25).square()),
                               exact_even_product=summary(even), exact_odd_product=summary(odd),
                               odd_to_even_energy=float(odd.square().sum()/even.square().sum().clamp_min(1e-30)),
                               decomposition_max_error=float((actual-even-odd).abs().max()))
        synthetic = dict(label='Counterfactual normalized-observation sign/radius interventions; these are not physics rollouts.',
                         sign_first_cosine=summary(F.cosine_similarity(features[:, 1], sign_features[:, 1], dim=-1)),
                         sign_final_cosine=summary(F.cosine_similarity(features[:, -1], sign_features[:, -1], dim=-1)),
                         radial={})
        for scale in (.01, .5, 1.01, 2.):
            changed = stages(agent.actor, scale * obs)
            synthetic['radial'][str(scale)] = dict(
                first_cosine=summary(F.cosine_similarity(features[:, 1], changed[:, 1], dim=-1)),
                final_cosine=summary(F.cosine_similarity(features[:, -1], changed[:, -1], dim=-1)),
                logit_change_rms=summary(agent.actor(scale*obs)-agent.actor(obs)))
    feature_jacobian = torch.compile(
        torch.func.vmap(torch.func.jacrev(lambda x: stages(agent.actor, x))),
        fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    jacobians = feature_jacobian(jacobian_obs).detach()
    jacobian_stats = {name: spectra(jacobians[:, i]) for i, name in enumerate(names)}
    # JVP along radius compared with total input Jacobian sensitivity.
    radial_derivative = torch.einsum('nsij,nj->nsi', jacobians, jacobian_obs)
    radial_stats = {name: summary(radial_derivative[:, i].norm(dim=-1)) for i, name in enumerate(names)}
    normalization = {}
    with torch.no_grad():
        selected = stages(agent.actor, jacobian_obs)
        for name, raw_jac, unit_jac, raw in (
                ('first', jacobians[:, 0], jacobians[:, 1], selected[:, 0]),
                ('branch', jacobians[:, 2], jacobians[:, 3], selected[:, 2]),
                ('residual_sum', jacobians[:, 1]+jacobians[:, 3], jacobians[:, 4],
                 selected[:, 1]+selected[:, 3])):
            direction = F.normalize(raw, dim=-1)
            component = torch.einsum('ni,nij->nj', direction, raw_jac)
            expected = (raw_jac-direction[..., None]*component[:, None]) / raw.norm(dim=-1)[:, None, None].clamp_min(1e-12)
            normalization[name] = dict(
                radial_feature_derivative_energy_removed=summary(component.square().sum(-1)
                    / raw_jac.square().sum((-2,-1)).clamp_min(1e-30)),
                normalized_to_raw_jacobian_norm=summary(unit_jac.norm(dim=(-2,-1))
                    / raw_jac.norm(dim=(-2,-1)).clamp_min(1e-30)),
                analytical_jacobian_relative_error=summary((unit_jac-expected).norm(dim=(-2,-1))
                    / unit_jac.norm(dim=(-2,-1)).clamp_min(1e-30)))
    def head_coordinates(x):
        alpha, beta, c, mean, _, _ = beta_quantities(agent, x)
        return torch.stack((mean, c.log()))
    policy_jacobian = torch.compile(torch.func.vmap(torch.func.jacrev(head_coordinates)),
                                    fullgraph=True, dynamic=False,
                                    options={"triton.cudagraphs": False})
    policy_jac = policy_jacobian(jacobian_obs).detach()
    mean_jac, concentration_jac = policy_jac[:, 0], policy_jac[:, 1]
    return dict(input_norm=summary(obs.norm(dim=-1)), input_coordinates=summary(obs),
                features=feature_stats, gates=gates, synthetic_interventions=synthetic,
                input_feature_jacobians=jacobian_stats, radial_JVP_norm=radial_stats,
                normalization_derivative_geometry=normalization,
                policy_mean_input_jacobian=spectra(mean_jac),
                policy_log_concentration_input_jacobian=spectra(concentration_jac),
                mean_vs_log_concentration_input_cosine=summary(F.cosine_similarity(mean_jac, concentration_jac, dim=-1)),
                mean_vs_log_concentration_norm_ratio=summary(mean_jac.norm(dim=-1)
                    / concentration_jac.norm(dim=-1).clamp_min(1e-20)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', action='append', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--module', default='cleanrl.ppo_continuous_action_sepclip_anvil2_v4')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--jacobian-samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if min(args.samples, args.jacobian_samples) < 1:
        raise ValueError('sample counts must be positive')
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    module = importlib.import_module(args.module)
    result = dict(exact_identities=[
        'SiTU(x) even component is .5*cap(g)*cap(u); odd component is even*tanh(g/2), for bias-free gate/up.',
        'For alpha,beta>=1, |physical mean| <= 1-2/concentration.',
        'Normalization Jacobian removes the radial feature direction; input radius need not be removed for nonlinear features.'
    ], limits=[
        'Synthetic sign and radial states need not be visited or physically valid.',
        'Jacobian spectra measure local sensitivity, not attainable return or a hard capacity ceiling.',
        'Torque proportions do not prove an optimal policy requires actuator saturation.',
        'Modes are undefined for numerically uniform Beta(alpha=beta=1) rows; mode summaries and torque proportions omit those rows.',
        'Initialization statistics use visited observations from a trained rollout, not an initial rollout distribution.',
        'No action-counterfactual physics returns are collected here.'
    ], hypotheses_to_falsify=[
        'If visited first features are dominated by even components and have low sign/radial sensitivity, a linear observation route or bias may recover useful velocity information.',
        'If alpha/beta rows align and policy mean sensitivity is much lower than concentration sensitivity, a location/precision parameterization may improve steering.',
        'If sampled torques rarely approach bounds while mode torques do, mean-mode physics evaluation can distinguish stochastic execution from representation limitations.'
    ], snapshots=[])
    for path in args.snapshot:
        snapshot = torch.load(path, map_location='cuda', weights_only=False)
        batch_obs = snapshot['batch']['obs']
        generator = torch.Generator(device='cuda').manual_seed(args.seed + 81829)
        indices = torch.randperm(batch_obs.shape[0], device='cuda', generator=generator)[:args.samples]
        obs = batch_obs[indices]
        agent = make_agent(module, snapshot)
        agent.load_state_dict(snapshot['model'])
        row = dict(snapshot=str(path), step=snapshot['global_step'], seed=args.seed,
                   observations=len(obs), jacobian_observations=min(args.jacobian_samples, len(obs)),
                   initial=initialized_report(module, snapshot, obs, args.seed),
                   trained_head=head_report(agent, obs, args.seed),
                   visited_representation=visited_report(agent, obs, obs[:args.jacobian_samples]))
        result['snapshots'].append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(snapshot=str(path), step=row['step'], input_norm=row['visited_representation']['input_norm'],
                              concentration=row['trained_head']['concentration'])), flush=True)


if __name__ == '__main__':
    main()

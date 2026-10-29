"""CUDA probe contract; execute through mlq. Static parity checks need no accelerator."""
import ast
from pathlib import Path

import pytest
import torch

from cleanrl.reppo import ppo_continuous_action_reppo_latentroll_diagnostic_v29 as diagnostic

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_learner_functions_unchanged():
    def functions(path):
        tree = ast.parse(path.read_text())
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    baseline = functions(ROOT / 'cleanrl/reppo/ppo_continuous_action_reppo_latentroll_v27.py')
    instrumented = functions(Path(diagnostic.__file__))
    for name in ('Actor', 'Critic', 'actor_loss', 'critic_loss', 'critic_step', 'actor_step',
                 'refresh_targets', 'bootstrap_values', 'uniform_rows', 'critic_diagnostics'):
        assert baseline[name] == instrumented[name], name


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; queue with mlq')
def test_observational_probe_preserves_rng_parameters_and_gradients():
    torch.manual_seed(1)
    args = diagnostic.Args(critic_hidden_dim=32, num_bins=31)
    critic = diagnostic.Critic(5, 2, args, 'cuda').cuda()
    import copy
    target = copy.deepcopy(critic).requires_grad_(False)
    rows, horizon = 16, 3
    obs, actions = torch.randn(rows, 5, device='cuda'), torch.randn(rows, 2, device='cuda').tanh()
    targets = torch.randn(rows, device='cuda')
    later_obs = torch.randn(horizon, rows, 5, device='cuda')
    later_actions = torch.randn(horizon, rows, 2, device='cuda').tanh()
    # Some episodes end immediately, others survive one/two/all transitions.
    survival_lengths = torch.arange(rows, device='cuda') % (horizon + 1)
    chain = (torch.arange(1, horizon + 1, device='cuda')[:, None] <= survival_lengths).float()
    rewards = torch.randn(horizon, rows, device='cuda')
    for parameter in critic.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: value.clone() for name, value in critic.state_dict().items()}
    grads = [p.grad.clone() for p in critic.parameters()]
    cuda_rng, cpu_rng = torch.cuda.get_rng_state().clone(), torch.get_rng_state().clone()
    metrics = diagnostic.observational_critic_probe(critic, target, obs, actions, targets,
        later_obs, later_actions, chain, rewards, 1., 0.)
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    for name, value in critic.state_dict().items():
        assert torch.equal(before[name], value)
    for old, parameter in zip(grads, critic.parameters()):
        assert torch.equal(old, parameter.grad)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics['probe/encoder_reward_weighted_grad_norm'] == 0
    assert metrics['probe/encoder_reward_grad_norm'] > 0
    assert all(not value.requires_grad for value in metrics.values())

    # Independent reference: index valid entries directly instead of multiplying masks.
    with torch.no_grad():
        features = critic.features(obs, actions)
        labels = critic.hl.project(targets)
        reference_ce = -(labels * critic.logits(features).log_softmax(-1)).sum(-1).mean()
        predicted = [features]
        for k in range(horizon):
            predicted.append(critic.transition(predicted[-1], later_actions[k]))
        predicted = torch.stack(predicted)
        target_features = target.features(later_obs, later_actions)
        latent_errors = (predicted[1:] - target_features).square().mean(-1)
        reference_aux = latent_errors[chain.bool()].mean()
        valid_rewards = torch.cat((torch.ones_like(chain[:1], dtype=torch.bool), chain[:-1].bool()))
        reward_errors = (critic.reward(predicted[:-1]) - rewards).square()
        reference_reward = reward_errors[valid_rewards].mean()
    torch.testing.assert_close(metrics['probe/ce_loss'], reference_ce)
    torch.testing.assert_close(metrics['probe/latent_loss'], reference_aux)
    torch.testing.assert_close(metrics['probe/reward_loss'], reference_reward)
    for k in range(horizon):
        valid = chain[k].bool()
        reference_mse = latent_errors[k, valid].mean()
        reference_energy = target_features[k, valid].square().mean()
        torch.testing.assert_close(metrics[f'probe/h{k + 1}_latent_mse'], reference_mse)
        torch.testing.assert_close(metrics[f'probe/h{k + 1}_normalized_latent_mse'],
                                   reference_mse / reference_energy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; queue with mlq')
def test_snapshot_state_has_independent_cpu_storage():
    args = diagnostic.Args(actor_hidden_dim=32)
    actor = diagnostic.Actor(5, 2, args).cuda()
    snapshot = diagnostic.cpu_state_dict(actor.net)
    expected = {name: value.clone() for name, value in snapshot.items()}
    assert all(value.device.type == 'cpu' for value in snapshot.values())
    with torch.no_grad():
        for parameter in actor.net.parameters():
            parameter.add_(1)
    for name, value in snapshot.items():
        assert torch.equal(value, expected[name])
    # Calling on an already-host snapshot must clone too: detach().cpu() alone aliases.
    host = torch.nn.Linear(2, 2)
    host_snapshot = diagnostic.cpu_state_dict(host)
    with torch.no_grad():
        host.weight.add_(1)
    assert not torch.equal(host_snapshot['weight'], host.weight)

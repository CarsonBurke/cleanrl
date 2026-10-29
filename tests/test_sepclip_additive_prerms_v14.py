"""Architecture/control CUDA contracts; run through mlq, never as bare ML."""
import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_sepclip_additive_prerms_anvil_v14 as trial
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as control
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions_host

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.fixture(autouse=True)
def runtime():
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def spaces():
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)),
                           single_action_space=gym.spaces.Box(-1., 1., (6,)))


def args(kind, architecture='additive_prerms'):
    result = trial.validate_args(trial.Args(actor_optimizer=kind, actor_architecture=architecture,
                                           num_steps=8, update_epochs=4, compile=True))
    result.num_iterations = 4
    return result


def compile_function(function):
    return torch.compile(function, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})


def batch(agent):
    obs = torch.randn(128, 17, device='cuda')
    native = torch.rand(128, 6, device='cuda').clamp(.01, .99)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(obs)
        values = values.flatten().clone()
        logp = agent.action_logprob(alpha, beta, native).clone()
    adv = torch.randn(128, device='cuda')
    return obs, native, logp, adv, values+adv, values


def learner(agent, module, config):
    return compile_function(lambda x, n, lp, adv, ret, val:
                            module.ppo_loss(agent, x, n, lp, adv, ret, val, config))


def update(agent, loss_fn, data, matrix, auxiliary):
    agent.zero_grad(set_to_none=True)
    loss, _ = loss_fn(*data)
    loss.backward()
    for trunk in (agent.actor, agent.critic):
        torch.nn.utils.clip_grad_norm_(tuple(trunk.parameters()), .5, foreach=True)
    if matrix is not None: matrix.step()
    auxiliary.step()
    agent.normalize_matrices()


@pytest.mark.parametrize('kind', ['adam', 'anvil2'])
def test_original_architecture_matches_v12_policy_rng_and_compiled_updates(kind):
    config = args(kind, 'original')
    torch.manual_seed(1)
    new = trial.Agent(spaces(), actor_architecture='original').cuda()
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(1)
    old = control.Agent(spaces()).cuda()
    torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
    new.normalize_matrices(); old.normalize_matrices()
    for key, value in new.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[key], atol=0, rtol=0)
    new_matrix, new_adam = trial.make_optimizers(new, config)
    old_matrix, old_adam = control.make_optimizers(old, config)
    if new_matrix is not None:
        new_matrix.begin_rollout(4); old_matrix.begin_rollout(4)
    data = batch(new)
    funcs = learner(new, trial, config), learner(old, control, config)
    for _ in range(4):
        update(new, funcs[0], data, new_matrix, new_adam)
        update(old, funcs[1], data, old_matrix, old_adam)
        for key, value in new.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[key], atol=0, rtol=0)


def test_additive_signed_anchor_retains_amplitude_and_pre_rms_residual_identity():
    torch.manual_seed(5)
    agent = trial.Agent(spaces()).cuda()
    agent.normalize_matrices()
    trunk = agent.actor.trunk
    obs = torch.randn(32, 17, device='cuda')
    # Independent composition proves pre-RMS occurs inside each branch, while
    # the identity stream and its final amplitude are retained.
    h = trunk.in_proj(obs)
    for index, block in enumerate(trunk.blocks):
        branch_input = torch.nn.functional.rms_norm(h, (64,), eps=1e-5)
        h = h + trunk.block_gates[index].sigmoid()*block(branch_input)
    torch.testing.assert_close(compile_function(trunk)(obs), h, atol=3e-6, rtol=3e-6)
    with torch.no_grad():
        trunk.in_proj.bias.zero_()
        for block in trunk.blocks: block.down.weight.zero_()
    fn = compile_function(trunk)
    torch.testing.assert_close(fn(obs), trunk.in_proj(obs), atol=0, rtol=0)
    torch.testing.assert_close(fn(-2*obs), -2*fn(obs), atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['adam', 'anvil2'])
def test_additive_gradients_optimizer_partition_and_retraction_controls(kind):
    torch.manual_seed(7)
    agent = trial.Agent(spaces()).cuda()
    config = args(kind)
    actor_body = trial.hidden_matrix_parameters(agent.actor)
    before = [p.detach().clone() for p in actor_body]
    agent.normalize_matrices()
    for p, saved in zip(actor_body, before): torch.testing.assert_close(p, saved, atol=0, rtol=0)
    matrix, auxiliary = trial.make_optimizers(agent, config)
    matrix_ids = set() if matrix is None else {id(p) for g in matrix.param_groups for p in g['params']}
    auxiliary_ids = {id(p) for g in auxiliary.param_groups for p in g['params']}
    assert not matrix_ids & auxiliary_ids
    assert matrix_ids | auxiliary_ids == {id(p) for p in agent.parameters()}
    if matrix is not None:
        assert matrix_ids == {id(p) for p in actor_body}
        assert id(agent.actor.trunk.in_proj.weight) in matrix_ids
        assert id(agent.actor.trunk.in_proj.bias) in auxiliary_ids
        assert all(id(p) in auxiliary_ids for p in agent.actor.trunk.block_gates)
        assert all(id(p) in auxiliary_ids for p in agent.critic.parameters())
        assert all(id(p) in auxiliary_ids for p in agent.actor.head.parameters())
    obs = torch.randn(32, 17, device='cuda')
    weights = torch.randn(32, 12, device='cuda')
    (compile_function(agent.actor)(obs)*weights).sum().backward()
    for parameter in (agent.actor.trunk.in_proj.weight, agent.actor.trunk.in_proj.bias,
                      *agent.actor.trunk.block_gates):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0
    alpha, beta, values = agent.get_policy_and_value(obs)
    expected = 1+torch.nn.functional.softplus(agent.actor(obs))
    torch.testing.assert_close(torch.cat((alpha, beta), -1), expected, atol=0, rtol=0)
    torch.testing.assert_close(values, agent.critic(obs), atol=0, rtol=0)
    torch.testing.assert_close(agent.actor.head[0].weight.norm(dim=1), torch.ones(12, device='cuda'), atol=2e-7, rtol=2e-7)


def test_compiled_additive_host_beta_rng_and_permanent_buffer_staging():
    torch.manual_seed(11)
    agent = trial.Agent(spaces()).cuda(); agent.normalize_matrices()
    mirror = trial.AdditiveHostMirror(agent.actor, 16)
    assert mirror.fused, mirror.fallback_reason
    device_actor = compile_function(agent.actor)
    sampler = make_beta_sampler(16, 6, np.full(6, -1., np.float32), np.ones(6, np.float32))
    assert sampler.fused, sampler.fallback_reason
    transfer = RolloutTransfer(2, 16, (17,), 'cuda', fields={'observations': (17,), 'native_actions': (6,)})
    rng, reference_rng = np.random.default_rng(91), np.random.default_rng(91)
    observations = np.random.default_rng(92).standard_normal((2, 16, 17)).astype(np.float32)
    saved_native, buffer_id = [], None
    for step in range(2):
        logits = mirror(observations[step])
        if buffer_id is None: buffer_id = id(logits)
        else: assert id(logits) == buffer_id
        np.testing.assert_allclose(logits, device_actor(torch.as_tensor(observations[step], device='cuda')).detach().cpu().numpy(), atol=3e-5, rtol=3e-5)
        expected_native, expected_physical = sample_beta_actions_host(logits, np.full(6, -1., np.float32), np.ones(6, np.float32), reference_rng)
        native, physical = sampler(logits, rng)
        np.testing.assert_array_equal(native, expected_native)
        np.testing.assert_array_equal(physical, expected_physical)
        assert rng.bit_generator.state == reference_rng.bit_generator.state
        saved_native.append(native.copy())
        transfer.push(step, np.zeros(16), np.zeros(16), np.zeros(16),
                      observations=observations[step], native_actions=native)
    sampler(np.zeros((16, 12), np.float32), rng)  # Overwrite permanent sampler buffers.
    got = transfer.upload()
    torch.testing.assert_close(got.fields['native_actions'], torch.as_tensor(np.stack(saved_native), device='cuda'), atol=0, rtol=0)
    torch.testing.assert_close(got.fields['observations'], torch.as_tensor(observations, device='cuda'), atol=0, rtol=0)
    old_logits = mirror(observations[0]).copy()
    with torch.no_grad(): agent.actor.readout_gain.mul_(1.7)
    np.testing.assert_array_equal(mirror(observations[0]), old_logits)
    mirror.refresh()
    np.testing.assert_allclose(mirror(observations[0]), device_actor(torch.as_tensor(observations[0], device='cuda')).detach().cpu().numpy(), atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize('kind', ['adam', 'anvil2'])
def test_compiled_additive_checkpoint_continues_exactly(kind):
    torch.manual_seed(13)
    config = args(kind)
    original = trial.Agent(spaces()).cuda(); original.normalize_matrices()
    matrix, auxiliary = trial.make_optimizers(original, config)
    if matrix is not None: matrix.begin_rollout(4)
    data = batch(original)
    original_loss = learner(original, trial, config)
    for _ in range(2): update(original, original_loss, data, matrix, auxiliary)
    resumed = trial.Agent(spaces()).cuda()
    resumed.load_state_dict(copy.deepcopy(original.state_dict()))
    new_matrix, new_auxiliary = trial.make_optimizers(resumed, config)
    new_auxiliary.load_state_dict(copy.deepcopy(auxiliary.state_dict()))
    if matrix is not None: new_matrix.load_state_dict(copy.deepcopy(matrix.state_dict()))
    new_loss = learner(resumed, trial, config)
    for _ in range(2):
        update(original, original_loss, data, matrix, auxiliary)
        update(resumed, new_loss, data, new_matrix, new_auxiliary)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
        for old_group, new_group in zip(auxiliary.param_groups, new_auxiliary.param_groups):
            for old_parameter, new_parameter in zip(old_group['params'], new_group['params']):
                for key, value in auxiliary.state[old_parameter].items():
                    torch.testing.assert_close(value, new_auxiliary.state[new_parameter][key], atol=0, rtol=0)
        if matrix is not None:
            for old_bank, new_bank in zip(matrix.stacks, new_matrix.stacks):
                assert new_bank[2] is new_matrix.state[new_bank[1][0]]
                for key in ('velocity', 'second'):
                    torch.testing.assert_close(old_bank[2][key], new_bank[2][key], atol=0, rtol=0)


def test_additive_choice_keeps_exact_original_critic_initialization_and_action_bounds():
    torch.manual_seed(19)
    additive = trial.Agent(spaces(), actor_architecture='additive_prerms').cuda()
    torch.manual_seed(19)
    original = control.Agent(spaces()).cuda()
    additive.normalize_matrices(); original.normalize_matrices()
    for key, value in additive.critic.state_dict().items():
        torch.testing.assert_close(value, original.critic.state_dict()[key], atol=0, rtol=0)
    for name in ('action_low', 'action_high', 'action_scale', 'log_action_scale'):
        torch.testing.assert_close(getattr(additive, name), getattr(original, name), atol=0, rtol=0)

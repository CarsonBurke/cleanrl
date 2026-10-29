"""CUDA contracts for intrinsic sphere transport; execute only in mlq jobs."""
import copy

import pytest
import torch

from cleanrl import ppo_continuous_action_sepclip_constraint_transport_v15 as trial
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as control
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.fixture(autouse=True)
def runtime():
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def compiled(function):
    return torch.compile(function, fullgraph=True, dynamic=False,
                         options={'triton.cudagraphs': False})


def normalized(shape, dim):
    return torch.nn.functional.normalize(torch.randn(*shape, device='cuda'), dim=dim)


@pytest.mark.parametrize('dim', [0, 1])
def test_finite_endpoint_parallel_transport_is_tangent_isometric_and_reversible(dim):
    torch.manual_seed(37)
    old = normalized((7, 5), dim)
    raw_a, raw_b = torch.randn_like(old), torch.randn_like(old)
    tangent = compiled(lambda p, v: trial.tangent_projection(p, v, dim))
    transport = compiled(lambda p, q, v: trial.sphere_parallel_transport(p, q, v, dim))
    a, b = tangent(old, raw_a), tangent(old, raw_b)
    # A real finite retraction, sufficiently far from old for projection alone
    # to lose norm, but in the non-antipodal hemisphere.
    new = torch.nn.functional.normalize(old + .7*a, dim=dim)
    saved_old, saved_new, saved_a = old.clone(), new.clone(), a.clone()
    ta, tb = transport(old, new, a), transport(old, new, b)
    torch.testing.assert_close((ta*new).sum(dim), torch.zeros_like((ta*new).sum(dim)), atol=8e-7, rtol=0)
    torch.testing.assert_close(ta.square().sum(dim), a.square().sum(dim), atol=3e-6, rtol=2e-6)
    torch.testing.assert_close((ta*tb).sum(dim), (a*b).sum(dim), atol=3e-6, rtol=2e-6)
    torch.testing.assert_close(transport(new, old, ta), a, atol=2e-6, rtol=2e-6)
    assert float((trial.tangent_projection(new, a).square().sum(dim)-a.square().sum(dim)).abs().max()) > .01
    # Transport must not overwrite endpoints or the caller-owned tangent.
    torch.testing.assert_close(old, saved_old, atol=0, rtol=0)
    torch.testing.assert_close(new, saved_new, atol=0, rtol=0)
    torch.testing.assert_close(a, saved_a, atol=0, rtol=0)
    torch.testing.assert_close(a, tangent(old, raw_a), atol=0, rtol=0)


@pytest.mark.parametrize('dim', [0, 1])
def test_pure_normal_force_cannot_generate_intrinsic_rotation(dim):
    # Nonorthogonal lanes reproduce the polar(normal-force) rotation witness.
    rows = torch.tensor([[1., 0.], [.6, .8]], device='cuda')
    p = torch.nn.Parameter(rows.clone() if dim == 1 else rows.T.contiguous())
    opt = trial.MatrixOptimizer([p], geometry='intrinsic', constraint_dims=[dim],
                                precision='fp32', total_steps=10, compile=True)
    before = p.detach().clone()
    opt.begin_rollout(3)
    for _ in range(3):
        p.grad = before.clone()
        saved_gradient = p.grad.clone()
        opt.step()
        torch.testing.assert_close(p, before, atol=2e-6, rtol=0)
        torch.testing.assert_close(p.grad, saved_gradient, atol=0, rtol=0)
        assert float(opt.state[p]['velocity'].abs().max()) < 2e-6


@pytest.mark.parametrize('pending', [False, True])
def test_compiled_checkpoint_preserves_previous_point_banks_and_rollout_phase(pending):
    torch.manual_seed(8)
    dims = [1, 1, 0]
    weights = [torch.nn.Parameter(normalized(shape, dim))
               for shape, dim in zip([(5, 7), (5, 7), (7, 5)], dims)]
    opt = trial.MatrixOptimizer(weights, geometry='intrinsic', constraint_dims=dims,
                                total_steps=12, precision='fp32', compile=True)
    opt.begin_rollout(4)
    for _ in range(2):
        for p in weights: p.grad = torch.randn_like(p)
        opt.step()
    for group, ps, state, _ in opt.stacks:
        # Base point belongs to this updated endpoint, not to a stale pre-step
        # parameter view or to the last independently normalized caller weight.
        torch.testing.assert_close(state['base_point'], torch.stack(ps), atol=2e-6, rtol=2e-6)
        for p in ps: assert state['base_point'].data_ptr() != p.data_ptr()
    if pending: opt.begin_rollout(4)
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    resumed = trial.MatrixOptimizer(clones, geometry='intrinsic', constraint_dims=dims,
                                    total_steps=12, precision='fp32', compile=True)
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert resumed.param_groups[0]['rollout_pending'] == pending
    for rate in (.0017, .0009):
        opt.set_lr(rate); resumed.set_lr(rate)
        gradients = []
        for p, q in zip(weights, clones):
            g = torch.randn_like(p); gradients.append(g.clone())
            p.grad, q.grad = g.clone(), g.clone()
        opt.step(); resumed.step()
        for p, q, g in zip(weights, clones, gradients):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
            torch.testing.assert_close(p.grad, g, atol=0, rtol=0)
        for a, b in zip(opt.stacks, resumed.stacks):
            assert b[2] is resumed.state[b[1][0]]
            for key in ('velocity', 'second', 'base_point', 'diagnostics'):
                torch.testing.assert_close(a[2][key], b[2][key], atol=0, rtol=0)


def test_rollout_slow_rail_ignores_later_epoch_credit_at_fixed_endpoint():
    torch.manual_seed(19)
    p = torch.nn.Parameter(normalized((5, 7), 1))
    opt = trial.MatrixOptimizer([p], geometry='intrinsic', constraint_dims=[1],
                                total_steps=12, precision='fp32', compile=True)
    opt.set_lr(0.)  # Isolate the clock from the separate transport effect.
    expected = torch.zeros_like(p)
    for _ in range(3):
        initial = torch.randn_like(p)
        projected = trial.tangent_projection(p, initial, 1)
        opt.begin_rollout(4)
        expected = .98**4*expected + (1-.98**4)*projected
        for epoch in range(4):
            p.grad = initial.clone() if epoch == 0 else torch.randn_like(p)*100
            opt.step()
            torch.testing.assert_close(opt.state[p]['velocity'][1, 0], expected, atol=2e-6, rtol=3e-6)
    assert opt.param_groups[0]['rollouts'] == 3
    assert opt.param_groups[0]['steps'] == 12


def test_ambient_same_source_control_preserves_legacy_matrix_updates():
    torch.manual_seed(31)
    weights = [torch.nn.Parameter(normalized(shape, dim))
               for shape, dim in [((5, 7), 1), ((7, 5), 0)]]
    clones = [torch.nn.Parameter(p.detach().clone()) for p in weights]
    new = trial.MatrixOptimizer(weights, geometry='ambient', total_steps=8, compile=True)
    old = control.MatrixOptimizer(clones, total_steps=8, compile=True)
    new.begin_rollout(4); old.begin_rollout(4)
    for _ in range(4):
        for p, q in zip(weights, clones):
            g = torch.randn_like(p); p.grad, q.grad = g.clone(), g.clone()
        new.step(); old.step()
        for p, q in zip(weights, clones): torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_geometry_mismatch_cannot_silently_reinterpret_checkpoint():
    p = torch.nn.Parameter(normalized((5, 7), 1))
    ambient = trial.MatrixOptimizer([p], geometry='ambient', compile=True)
    q = torch.nn.Parameter(p.detach().clone())
    intrinsic = trial.MatrixOptimizer([q], geometry='intrinsic', constraint_dims=[1], compile=True)
    with pytest.raises(ValueError): intrinsic.load_state_dict(copy.deepcopy(ambient.state_dict()))


def test_intrinsic_actor_matrix_step_preserves_head_critic_and_gradient_ownership():
    import gymnasium as gym
    import numpy as np
    from types import SimpleNamespace
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)),
                          single_action_space=gym.spaces.Box(-1., 1., (6,)))
    torch.manual_seed(43)
    agent = trial.Agent(env).cuda(); agent.normalize_matrices()
    config = trial.validate_args(trial.Args(matrix_geometry='intrinsic', compile=True))
    config.num_iterations = 3
    matrix, auxiliary = trial.make_optimizers(agent, config)
    selected = {id(p) for g in matrix.param_groups for p in g['params']}
    adam = {id(p) for g in auxiliary.param_groups for p in g['params']}
    assert not selected & adam
    assert selected | adam == {id(p) for p in agent.parameters()}
    expected_axes = {id(layer.weight): dim for stage in (agent.actor.first, agent.actor.second)
                     for layer, dim in ((stage[0].gate, 1), (stage[0].up, 1), (stage[0].down, 0))}
    assert selected == set(expected_axes)
    for group in matrix.param_groups:
        assert [expected_axes[id(p)] for p in group['params']] == group['constraint_dims']
    obs = torch.randn(32, 17, device='cuda')
    coefficient = torch.randn(32, 12, device='cuda')
    (compiled(agent.actor)(obs)*coefficient).sum().backward()
    protected = {name: p.detach().clone() for name, p in agent.named_parameters() if id(p) not in selected}
    gradients = {name: p.grad.clone() for name, p in agent.named_parameters() if p.grad is not None}
    matrix.begin_rollout(4); matrix.step()
    for name, p in agent.named_parameters():
        if name in protected: torch.testing.assert_close(p, protected[name], atol=0, rtol=0)
        if name in gradients: torch.testing.assert_close(p.grad, gradients[name], atol=0, rtol=0)
    for _, ps, state, _ in matrix.stacks:
        dim = state['constraint_dim']-2
        endpoint = torch.stack(ps)
        for rail in state['velocity']:
            torch.testing.assert_close((rail*endpoint).sum(dim), torch.zeros_like((rail*endpoint).sum(dim)), atol=3e-6, rtol=0)
        assert float(state['diagnostics'][0]) == 0
    # Reproduce the trainer's independent matrix normalization before the next
    # epoch; cached optimizer endpoints must not be aliased to caller weights.
    agent.normalize_matrices()
    for _, ps, state, _ in matrix.stacks:
        for p in ps: assert p.data_ptr() != state['base_point'].data_ptr()
    agent.zero_grad(set_to_none=True)
    (compiled(agent.actor)(obs)*coefficient).sum().backward()
    second_gradients = {name: p.grad.clone() for name, p in agent.named_parameters() if p.grad is not None}
    matrix.step()
    for name, p in agent.named_parameters():
        if name in second_gradients: torch.testing.assert_close(p.grad, second_gradients[name], atol=0, rtol=0)
    for _, ps, state, _ in matrix.stacks:
        endpoint = torch.stack(ps)
        axis = state['constraint_dim']-2
        torch.testing.assert_close(state['base_point'], endpoint, atol=0, rtol=0)
        for rail in state['velocity']:
            torch.testing.assert_close((rail*endpoint).sum(axis), torch.zeros_like((rail*endpoint).sum(axis)), atol=3e-6, rtol=0)
        assert float(state['diagnostics'][0]) == 0


def test_checkpoint_axis_metadata_is_not_inferred_from_matrix_shape():
    p = torch.nn.Parameter(normalized((5, 7), 1))
    first = trial.MatrixOptimizer([p], geometry='intrinsic', constraint_dims=[1], compile=True)
    q = torch.nn.Parameter(normalized((5, 7), 0))
    second = trial.MatrixOptimizer([q], geometry='intrinsic', constraint_dims=[0], compile=True)
    with pytest.raises(ValueError): second.load_state_dict(copy.deepcopy(first.state_dict()))

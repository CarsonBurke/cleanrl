"""Predictive actor code (mdl v8): v6 untouched without it, the replay and the exact function-space code."""

import types

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Beta

from cleanrl.mdl import ppo_continuous_action_mdl_critic_predictive_v6 as v6
from cleanrl.mdl import ppo_continuous_action_mdl_predictive_actor_v8 as v8

OBS, ACT, ROWS = 5, 3, 64
SLOTS, STATES = 4, 16


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(module, posterior="none", log_sigma2=-12.0):
    torch.manual_seed(0)
    agent = module.Agent(envs(), posterior, init_log_sigma2=log_sigma2)
    agent.normalize_matrices()
    return agent


def args_for(module, posterior="none", actor_code="none", **kwargs):
    args = module.Args(norm_adv=False, clip_vloss=False, clip_coef_upper=0.2, posterior=posterior, **kwargs)
    if actor_code != "none":
        args.actor_code = actor_code
    args.batch_size = ROWS
    return args


def data():
    torch.manual_seed(1)
    obs = torch.randn(ROWS, OBS)
    native = torch.rand(ROWS, ACT).clamp(1e-3, 1 - 1e-3)
    return obs, native, torch.randn(ROWS), torch.randn(ROWS), torch.randn(ROWS), torch.zeros(ROWS)


def make_replay(gamma=0.9, seed=4):
    return v8.ActorReplay(SLOTS, STATES, OBS, ACT, gamma, torch.device("cpu"), torch.Generator().manual_seed(seed))


def filled_replay(agent, rollouts=3, iteration=4):
    replay = make_replay()
    torch.manual_seed(5)
    for index in range(rollouts):
        replay.insert(torch.randn(ROWS, OBS), iteration - rollouts + index)
    replay.refresh(agent, iteration)
    return replay


def perturb_actor(agent, scale, seed=7):
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(scale * torch.randn_like(parameter))


@pytest.mark.parametrize("posterior", ["none", "mdl"])
def test_without_the_code_the_loss_is_v6(posterior):
    old, new = make_agent(v6, posterior), make_agent(v8, posterior)
    assert old.state_dict().keys() == new.state_dict().keys()
    new.load_state_dict(old.state_dict())
    kl_weight = torch.tensor(0.3)
    replay = filled_replay(new)
    torch.manual_seed(2)
    want = v6.ppo_loss(old, *data(), args_for(v6, posterior), kl_weight)
    torch.manual_seed(2)
    got = v8.ppo_loss(new, *data(), args_for(v8, posterior), kl_weight, torch.tensor(5.0), replay.tensors())
    torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
    torch.testing.assert_close(got[1][:6], want[1], rtol=0, atol=0)
    assert float(got[1][6]) == 0.0


def test_code_adds_the_weighted_exact_kl():
    agent = make_agent(v8)
    replay = filled_replay(agent)
    perturb_actor(agent, 0.05)
    args, weight = args_for(v8, actor_code="predictive"), torch.tensor(0.7)
    base = v8.ppo_loss(agent, *data(), args_for(v8), None, weight, replay.tensors())[0]
    coded = v8.ppo_loss(agent, *data(), args, None, weight, replay.tensors())[0]
    alpha, beta = (F.softplus(agent.actor(replay.observations)) + 1.0).chunk(2, dim=-1)
    per_slot = torch.distributions.kl_divergence(Beta(replay.alpha, replay.beta), Beta(alpha, beta)).sum(-1).mean(-1)
    want = (per_slot * replay.weights).sum()
    assert float(want.detach()) > 0
    torch.testing.assert_close(coded - base, weight * want, rtol=1e-5, atol=1e-7)
    metrics = v8.ppo_loss(agent, *data(), args, None, weight, replay.tensors())[1]
    torch.testing.assert_close(metrics[6], want.detach(), rtol=1e-6, atol=0)


def test_code_is_zero_with_zero_gradient_at_the_reference():
    agent = make_agent(v8)
    replay = filled_replay(agent)
    code = v8.actor_code(agent, *replay.tensors())
    grads = torch.autograd.grad(code, list(agent.actor.parameters()))
    assert float(code.detach().abs()) < 1e-6
    assert max(float(g.abs().max()) for g in grads) < 1e-5


def test_code_is_second_order_in_the_update():
    # Near the reference the code is 1/2 d^T F d: quadrupling as the step doubles, the Laplace limit.
    agent = make_agent(v8)
    replay = filled_replay(agent)
    state = {k: v.clone() for k, v in agent.state_dict().items()}
    codes = []
    for scale in (1e-3, 2e-3):
        agent.load_state_dict(state)
        perturb_actor(agent, scale)
        with torch.no_grad():
            codes.append(float(v8.actor_code(agent, *replay.tensors())))
    assert codes[1] / codes[0] == pytest.approx(4.0, rel=0.02)


def test_code_charges_only_the_actor():
    agent = make_agent(v8, "mdl")
    replay = filled_replay(agent)
    perturb_actor(agent, 0.05)
    v8.actor_code(agent, *replay.tensors()).backward()
    assert all(p.grad is None for p in agent.critic.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())


def test_refresh_weights_slots_by_age():
    agent = make_agent(v8)
    replay = make_replay(gamma=0.5)
    replay.refresh(agent, 1)
    assert torch.equal(replay.weights, torch.zeros(SLOTS))
    for iteration in (1, 2):
        replay.insert(torch.randn(ROWS, OBS), iteration)
    replay.refresh(agent, 3)
    torch.testing.assert_close(replay.weights, torch.tensor([0.25, 0.5, 0.0, 0.0]))
    alpha, beta = (F.softplus(agent.actor(replay.observations)) + 1.0).chunk(2, dim=-1)
    torch.testing.assert_close(replay.alpha, alpha, rtol=0, atol=0)
    torch.testing.assert_close(replay.beta, beta, rtol=0, atol=0)


def test_insert_fills_empty_slots_then_replaces_the_oldest():
    replay = make_replay()
    batches = [torch.arange(ROWS * OBS, dtype=torch.float32).view(ROWS, OBS) + 1000 * i for i in range(SLOTS + 2)]
    for iteration, batch in enumerate(batches, start=1):
        replay.insert(batch, iteration)
    assert replay.filled_at.tolist() == [5, 6, 3, 4]
    for slot, iteration in enumerate(replay.filled_at.tolist()):
        rows = replay.observations[slot]
        source = batches[iteration - 1]
        # Distinct rows of that rollout, not a prefix.
        matches = (rows[:, None, :] == source[None]).all(-1)
        assert matches.sum(-1).eq(1).all()
        assert matches.any(0).sum() == STATES
        assert not torch.equal(rows, source[:STATES])


def test_replay_uses_only_its_generator():
    agent = make_agent(v8)
    first = filled_replay(agent)
    torch.manual_seed(5)
    rollouts = [torch.randn(ROWS, OBS) for _ in range(3)]
    state = torch.get_rng_state()
    second = make_replay()
    for index, rollout in enumerate(rollouts):
        second.insert(rollout, 1 + index)
    second.refresh(agent, 4)
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(second.observations, first.observations, rtol=0, atol=0)


def test_beta_kl_matches_torch():
    torch.manual_seed(3)
    a, b, c, d = (1.0 + 20.0 * torch.rand(256, dtype=torch.float64) for _ in range(4))
    want = torch.distributions.kl_divergence(Beta(a, b), Beta(c, d))
    torch.testing.assert_close(v8.beta_kl(a, b, c, d), want, rtol=1e-10, atol=1e-12)


def test_compiled_loss_sees_in_place_replay_updates():
    torch.distributions.kl._KL_MEMOIZE.clear()  # the run compiles before any eager dispatch is memoised
    torch._dynamo.reset()
    agent = make_agent(v8)
    args = args_for(v8, actor_code="predictive")
    weight = torch.tensor(0.5)
    replay = filled_replay(agent)
    compiled = torch.compile(lambda *inputs: v8.ppo_loss(agent, *inputs[:6], args, None, inputs[6], inputs[7:])[0],
                             fullgraph=True, dynamic=False)
    for step in range(3):
        perturb_actor(agent, 0.02, seed=step)
        with torch.no_grad():
            weight.fill_(0.5 * (step + 1))
            replay.insert(torch.randn(ROWS, OBS), 4 + step)
        replay.refresh(agent, 5 + step)
        perturb_actor(agent, 0.02, seed=10 + step)
        got = compiled(*data(), weight, *replay.tensors())
        want = v8.ppo_loss(agent, *data(), args, None, weight, replay.tensors())[0]
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_validate_args():
    for kwargs in ({"actor_code": "laplace"}, {"actor_code": "predictive", "actor_temperature": 0.0},
                   {"actor_code": "predictive", "actor_forgetting": 1.0},
                   {"actor_code": "predictive", "replay_rollouts": 0},
                   {"actor_code": "predictive", "replay_states": 0},
                   {"actor_code": "predictive", "norm_adv": True}):
        with pytest.raises(ValueError):
            v8.validate_args(v8.Args(**{"norm_adv": False, **kwargs}))
    v8.validate_args(v8.Args(actor_code="predictive", norm_adv=False))


def test_telemetry_reads_the_newest_slot_and_leaves_state_alone():
    agent = make_agent(v8)
    replay = filled_replay(agent)
    obs, native, logprobs, advantages = data()[:4]
    with torch.no_grad():
        old_alpha, old_beta = (F.softplus(agent.actor(obs)) + 1.0).chunk(2, dim=-1)
    perturb_actor(agent, 0.05)
    args = args_for(v8, actor_code="predictive")
    state = torch.get_rng_state()
    with torch.no_grad():
        out = v8.actor_telemetry(agent, obs, native, logprobs, advantages, old_alpha, old_beta, replay,
                                 torch.tensor(0.5), args)
    assert torch.equal(torch.get_rng_state(), state)
    assert all(p.grad is None for p in agent.parameters())
    assert all(torch.isfinite(v).all() for v in out.values())
    per_slot = v8.replay_kl(agent, replay.observations, replay.alpha, replay.beta)
    newest = int(torch.argmax(replay.filled_at))
    torch.testing.assert_close(out["actor/replay_kl_newest"], per_slot[newest].detach())
    torch.testing.assert_close(out["actor/code_nats"], (per_slot * replay.weights).sum().detach())
    # The code's gradient is the weight times the code's own gradient.
    code = torch.autograd.grad(v8.actor_code(agent, *replay.tensors()), list(agent.actor.parameters()))
    norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in code]))
    torch.testing.assert_close(out["grad/actor_code_norm"], 0.5 * norm, rtol=1e-5, atol=0)

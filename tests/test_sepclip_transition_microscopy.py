"""CUDA contracts for observational parity and exact physics restoration; mlq only."""
from types import SimpleNamespace
import copy
import gymnasium as gym
import mujoco
import numpy as np
import pytest
import torch
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as new
from cleanrl import ppo_continuous_action_sepclip_rollout_clock_anvil_v7 as old
from cleanrl.awr import ppo_continuous_action_ppo_sepclip_control_v1 as adam_ref
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


def spaces():
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)),
        single_action_space=gym.spaces.Box(-1., 1., (6,)))


@pytest.mark.parametrize('kind', ['anvil2', 'adam'])
def test_compiled_learner_matches_frozen_control(kind):
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    reference = old if kind == 'anvil2' else adam_ref
    args = new.validate_args(new.Args(actor_optimizer=kind, num_steps=128, update_epochs=2))
    args.num_iterations = 10
    torch.manual_seed(1)
    a = new.Agent(spaces()).cuda()
    torch.manual_seed(1)
    b = reference.Agent(spaces()).cuda()
    a.normalize_matrices(); b.normalize_matrices()
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
    matrix_a, opt_a = new.make_optimizers(a, args)
    if kind == 'anvil2':
        matrix_b, opt_b = old.make_optimizers(b, args)
        matrix_a.begin_rollout(2); matrix_b.begin_rollout(2)
    else:
        matrix_b = None
        opt_b = torch.optim.Adam(b.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    obs = torch.randn(128, 17, device='cuda')
    native = torch.rand(128, 6, device='cuda').clamp(.01, .99)
    with torch.no_grad():
        alpha, beta, val = a.get_policy_and_value(obs)
        val = val.flatten().clone()
        logp = a.action_logprob(alpha, beta, native).clone()
    advantage = torch.randn(128, device='cuda')
    returns = val + advantage
    losses = [torch.compile(lambda x, n, lp, adv, ret, v, agent=agent, module=module:
        module.ppo_loss(agent, x, n, lp, adv, ret, v, args), fullgraph=True,
        options={'triton.cudagraphs': False}) for agent, module in ((a, new), (b, reference))]
    for _ in range(2):
        for agent, fn, matrix, opt in zip((a,b), losses, (matrix_a,matrix_b), (opt_a,opt_b)):
            loss, _ = fn(obs, native, logp, advantage, returns, val)
            agent.zero_grad(set_to_none=True); loss.backward()
            for trunk in (agent.actor, agent.critic):
                torch.nn.utils.clip_grad_norm_(tuple(trunk.parameters()), .5, foreach=True)
            if matrix is not None: matrix.step()
            opt.step(); agent.normalize_matrices()
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)


def test_capture_is_read_only_and_physics_roundtrip():
    configure_runtime()
    envs = new.make_mujoco_vector_env('HalfCheetah-v4', 2, num_threads=2)
    try:
        raw, _ = envs.reset(seed=1)
        obs_norm = VectorObsNorm(2, (17,)); rew_norm = VectorRewardNorm(2, .99)
        obs_norm.normalize(raw)
        capture = new.TransitionCapture(envs, 1)
        states = copy.deepcopy(np.random.get_state())
        torch_state = torch.cuda.get_rng_state().clone()
        norm_state = {k:getattr(obs_norm,k).copy() for k in ('means','variances','counts')}
        action = np.full((2, 6), .3, np.float32)
        capture.before(0, envs, action, obs_norm)
        for k,v in norm_state.items(): np.testing.assert_array_equal(v,getattr(obs_norm,k))
        np.testing.assert_array_equal(states[1],np.random.get_state()[1])
        torch.testing.assert_close(torch_state,torch.cuda.get_rng_state(),atol=0,rtol=0)
        next_raw, reward, terms, truncs, infos = envs.step(action)
        expected_raw, expected_rew = next_raw.copy(), reward.copy()
        normalized_reward = rew_norm.normalize(reward, terms)
        next_obs, transition_obs = obs_norm.normalize_step(next_raw, terms, truncs, infos)
        capture.after(0, next_raw, reward, next_obs, transition_obs, rew_norm)
        np.testing.assert_array_equal(normalized_reward,
            np.clip(capture.arrays['raw_rewards'][0]/capture.arrays['reward_std'][0],-10,10).astype(np.float32))
        for i, wrapper in enumerate(envs.envs):
            base=wrapper.unwrapped
            new.set_integration_state(base,capture.arrays['integration_states'][0,i])
            mujoco.mj_forward(base.model,base.data)
            new.set_integration_state(base,capture.arrays['integration_states'][0,i])
        restored_raw, restored_rew, *_ = envs.step(action)
        np.testing.assert_array_equal(restored_raw, expected_raw)
        np.testing.assert_array_equal(restored_rew, expected_rew)
    finally:
        envs.close()

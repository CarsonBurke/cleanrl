"""CPU contracts for 32x-LR/1-mb truegate and v30 target-critic ablations."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_denseres_v1 as dense_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_res_v1 as res_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_v1 as stiglu_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_stiglu_denseres_v1 as dense
from cleanrl import ppo_continuous_action_32xlr_1mb_stiglu_res_v1 as res
from cleanrl import ppo_continuous_action_32xlr_1mb_stiglu_v1 as stiglu
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_lrelusq_res_v1 as lrelures_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_lrelusq_res_v1 as lrelures
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_lrelusq_sphere_v1 as lrelusphere_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_v1 as sphere_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_lrelusq_res_v1 as lrelures_truegate_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_stiglu_sphere_1ep_v1 as sph1ep_noadv
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_stiglu_denseres_1ep_v1 as ep1
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_stiglu_sphere_1ep_v1 as sph1ep_tg
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_stiglu_sphere_v1 as sphere_truegate_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_stiglu_denseres_v1 as dense_truegate_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_stiglu_res_v1 as res_truegate_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_noadvnorm_stiglu_v1 as stiglu_truegate_noadvnorm
from cleanrl import ppo_continuous_action_32xlr_1mb_truegate_v1 as truegate
from cleanrl import ppo_continuous_action_32xlr_1mb_v1 as base
from cleanrl import ppo_continuous_action_32xlr_1mb_v30target_v1 as v30target_v1
from cleanrl import ppo_continuous_action_32xlr_1mb_v30target_v2 as v30target
from cleanrl.shared.host_actor import (
    LReluResTrunk, LReluSphereTrunk, LReluSqPair, SITU_GLU_MEAN_SQUARE,
    SiTUGLUBranch, SiTUDenseTrunk, SiTUResTrunk, SiTUSphereTrunk, justnorm,
)
from cleanrl.shared.ppo_loop import compute_gae_from_next_values


def test_32xlr_1mb_defaults():
    for module in (base, v30target_v1, v30target, truegate):
        args = module.Args()
        assert args.learning_rate == 9.6e-3
        assert args.num_minibatches == 1
    assert v30target_v1.Args().target_update_period == 100
    assert not hasattr(v30target.Args(), "target_update_period")


def test_truegate_uses_physical_logprob_without_subtracting_jacobian_again():
    advantages = torch.tensor([1.0, -1.0, 2.0])
    physical = torch.tensor([-1.0, -4.0, 0.0])
    weights = truegate.truegate_weights(advantages, physical, eta=1.0, surprisal_clip=10.0)
    surprisal = (-physical).clamp(-10.0, 10.0)
    expected = torch.sigmoid(advantages * surprisal)
    torch.testing.assert_close(weights, expected)
    # Double-subtracting the HalfCheetah logdet (~4.16) would move these weights.
    fake_native = physical + 4.16
    shifted = truegate.truegate_weights(advantages, fake_native, eta=1.0, surprisal_clip=10.0)
    assert not torch.allclose(weights, shifted)


def _stiglu_envs_stub():
    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

    return Stub()


def test_stiglu_preserves_optimizer_settings_and_advnorm_flags():
    assert stiglu.Args().learning_rate == 9.6e-3
    assert stiglu.Args().num_minibatches == 1
    assert stiglu.Args().norm_adv is True
    assert stiglu_noadvnorm.Args().norm_adv is False
    assert stiglu_truegate_noadvnorm.Args().norm_adv is False
    assert stiglu_noadvnorm.Args().learning_rate == 9.6e-3
    assert stiglu_truegate_noadvnorm.Args().num_minibatches == 1


def test_stiglu_trunks_stack_two_branches_with_no_legacy_activation():
    for module in (stiglu, stiglu_noadvnorm, stiglu_truegate_noadvnorm):
        torch.manual_seed(0)
        agent = module.Agent(_stiglu_envs_stub())
        branches = [m for m in agent.modules() if isinstance(m, SiTUGLUBranch)]
        assert len(branches) == 4, "two stacked branches in each of actor and critic"
        assert not any(isinstance(m, torch.nn.Tanh) for m in agent.modules())
        assert not any(type(m).__name__ == "LeakyReluSq" for m in agent.modules())
        assert not any(isinstance(m, torch.nn.ReLU) for m in agent.modules())
        assert branches[0].in_dim == 17 and branches[0].out_dim == 64
        assert branches[0].hidden_dim == round(2.0 * (64 + 1) / 3.0) == 43
        assert all(b.gate.bias is None and b.up.bias is None and b.down.bias is None for b in branches)


def test_stiglu_down_projection_matches_leakyrelusq_output_scale():
    torch.manual_seed(0)
    branch = stiglu.stiglu_branch(17, 64)
    expected_down_std = np.sqrt(6.375 * 64 / (branch.hidden_dim * SITU_GLU_MEAN_SQUARE))
    # torch orthogonal_ yields (semi-)orthonormal columns when rows >= cols, so
    # the gain spreads over the row count: down is 64x43 -> RMS = std/sqrt(64).
    assert branch.down.weight.pow(2).mean().sqrt().item() == pytest.approx(
        expected_down_std / np.sqrt(64), rel=0.05)
    # Gate is 43x17, likewise tall -> RMS = sqrt(2)/sqrt(43).
    assert branch.gate.weight.pow(2).mean().sqrt().item() == pytest.approx(
        np.sqrt(2) / np.sqrt(43), rel=0.05)


def test_res_preserves_optimizer_settings_and_advnorm_flags():
    assert res.Args().learning_rate == 9.6e-3
    assert res.Args().num_minibatches == 1
    assert res.Args().norm_adv is True
    assert res_noadvnorm.Args().norm_adv is False
    assert res_truegate_noadvnorm.Args().norm_adv is False
    assert hasattr(res_truegate_noadvnorm, "truegate_weights")


def test_res_trunks_have_gated_residuals_and_long_skip():
    for module in (res, res_noadvnorm, res_truegate_noadvnorm):
        torch.manual_seed(0)
        agent = module.Agent(_stiglu_envs_stub())
        trunks = [m for m in (agent.actor[0], agent.critic[0]) if isinstance(m, SiTUResTrunk)]
        assert len(trunks) == 2
        assert not any(type(m).__name__ == "LeakyReluSq" for m in agent.modules())
        for trunk in trunks:
            for gate in (trunk.lam1, trunk.lam2, trunk.lam_skip):
                assert gate.item() == pytest.approx(-1.5)
                assert gate.requires_grad
            assert trunk.block1.hidden_dim == trunk.block2.hidden_dim == 43
        # Unit-variance stream puts SiTU gate preacts at the v=2 design point,
        # and the trunk output stays in the tanh/lrelusq regime.
        trunk = agent.actor[0]
        xb = torch.randn(4096, 17)
        assert trunk.in_proj(xb).var().item() == pytest.approx(1.0, rel=0.1)
        assert trunk.block1.gate(trunk.in_proj(xb)).var().item() == pytest.approx(2.0, rel=0.15)
        assert trunk(xb).pow(2).mean().item() == pytest.approx(1.44, rel=0.2)


def test_dense_preserves_optimizer_settings_and_advnorm_flags():
    assert dense.Args().learning_rate == 9.6e-3
    assert dense.Args().num_minibatches == 1
    assert dense.Args().norm_adv is True
    assert dense_noadvnorm.Args().norm_adv is False
    assert dense_truegate_noadvnorm.Args().norm_adv is False
    assert hasattr(dense_truegate_noadvnorm, "truegate_weights")


def test_dense_trunks_wire_every_stream_to_every_later_stage():
    for module in (dense, dense_noadvnorm, dense_truegate_noadvnorm):
        torch.manual_seed(0)
        agent = module.Agent(_stiglu_envs_stub())
        trunks = [m for m in (agent.actor[0], agent.critic[0]) if isinstance(m, SiTUDenseTrunk)]
        assert len(trunks) == 2
        assert not any(type(m).__name__ == "LeakyReluSq" for m in agent.modules())
        for trunk in trunks:
            assert trunk.n_blocks == 3
            assert len(trunk.blocks) == 3
            # Dense: block k>1 receives k-1 skips; 3 skips total for 3 blocks.
            assert [k for k, _ in trunk.skip_index] == [2, 3, 3]
            assert [j for _, j in trunk.skip_index] == [0, 0, 1]
            gates = list(trunk.block_gates) + list(trunk.skip_gates)
            assert len(gates) == 6
            for gate in gates:
                # Per-perceptron: one learned value per channel, all small-init.
                assert tuple(gate.shape) == (64,)
                assert gate.requires_grad
                assert torch.allclose(gate, torch.full_like(gate, -1.5))
        trunk = agent.actor[0]
        xb = torch.randn(4096, 17)
        assert trunk(xb).pow(2).mean().item() == pytest.approx(2.47, rel=0.25)


def test_1ep_uses_one_actor_epoch_and_ten_critic_epochs():
    assert ep1.Args().actor_epochs == 1
    assert ep1.Args().critic_epochs == 10
    assert not hasattr(ep1.Args(), "update_epochs")
    assert ep1.Args().norm_adv is False
    assert hasattr(ep1, "truegate_weights")


def _1ep_agents():
    torch.manual_seed(0)
    a = ep1.Agent(_stiglu_envs_stub())
    torch.manual_seed(0)
    b = dense_truegate_noadvnorm.Agent(_stiglu_envs_stub())
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb)
    return a, b


def test_1ep_split_losses_sum_to_joint_ppo_loss():
    a, b = _1ep_agents()
    args = ep1.Args()
    obs = torch.randn(64, 17)
    native = torch.rand(64, 6).clamp(1e-6, 1 - 1e-6)
    old_logprobs = torch.randn(64)
    advantages = torch.randn(64) * 3.0
    returns = torch.randn(64) * 100.0
    old_values = torch.randn(64) * 100.0
    joint, _ = dense_truegate_noadvnorm.ppo_loss(b, obs, native, old_logprobs, advantages, returns, old_values, args)
    pa, _ = ep1.actor_loss(a, obs, native, old_logprobs, advantages, args)
    pv = ep1.critic_loss(a, obs, returns, old_values, args)
    torch.testing.assert_close(pa + pv, joint, rtol=1e-5, atol=1e-6)


def test_1ep_gradients_do_not_cross_actor_critic_boundary():
    a, _ = _1ep_agents()
    args = ep1.Args()
    obs = torch.randn(32, 17)
    native = torch.rand(32, 6).clamp(1e-6, 1 - 1e-6)
    ep1.actor_loss(a, obs, native, torch.randn(32), torch.randn(32), args)[0].backward()
    assert all(p.grad is None for p in a.critic.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.actor.parameters())
    a.zero_grad(set_to_none=True)
    ep1.critic_loss(a, obs, torch.randn(32), torch.randn(32), args).backward()
    assert all(p.grad is None for p in a.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.critic.parameters())


def test_sphere_preserves_optimizer_settings_and_advnorm_flags():
    assert sphere.Args().learning_rate == 9.6e-3
    assert sphere.Args().num_minibatches == 1
    assert sphere.Args().norm_adv is True
    assert sphere_noadvnorm.Args().norm_adv is False
    assert sphere_truegate_noadvnorm.Args().norm_adv is False
    assert hasattr(sphere_truegate_noadvnorm, "truegate_weights")


def test_sphere_trunks_live_on_the_unit_hypersphere():
    for module in (sphere, sphere_noadvnorm, sphere_truegate_noadvnorm):
        torch.manual_seed(0)
        agent = module.Agent(_stiglu_envs_stub())
        trunks = [m for m in (agent.actor[0], agent.critic[0]) if isinstance(m, SiTUSphereTrunk)]
        assert len(trunks) == 2
        assert not any(type(m).__name__ == "LeakyReluSq" for m in agent.modules())
        trunk = agent.actor[0]
        assert trunk.n_blocks == 3
        xb = torch.randn(4096, 17)
        out = trunk(xb)
        # Unit-hypersphere streams: every row has norm 1 by construction.
        assert torch.allclose(out.norm(p=2, dim=-1), torch.ones(4096), rtol=1e-4, atol=1e-6)
        # SiTU gate/up preacts sit at the v=2 design point on the stream.
        s0 = justnorm(trunk.in_proj(xb))
        assert trunk.blocks[0].gate(s0).var().item() == pytest.approx(2.0, rel=0.15)


def test_lrelures_preserves_optimizer_settings_and_advnorm_flags():
    assert lrelures.Args().learning_rate == 9.6e-3
    assert lrelures.Args().num_minibatches == 1
    assert lrelures.Args().norm_adv is True
    assert lrelures_noadvnorm.Args().norm_adv is False
    assert lrelures_truegate_noadvnorm.Args().norm_adv is False
    assert hasattr(lrelures_truegate_noadvnorm, "truegate_weights")


def test_lrelures_matches_res_scaffolding_with_pair_blocks():
    for module in (lrelures, lrelures_noadvnorm, lrelures_truegate_noadvnorm):
        torch.manual_seed(0)
        agent = module.Agent(_stiglu_envs_stub())
        trunks = [m for m in (agent.actor[0], agent.critic[0]) if isinstance(m, LReluResTrunk)]
        assert len(trunks) == 2
        assert not any(isinstance(m, SiTUGLUBranch) for m in agent.modules())
        for trunk in trunks:
            for gate in (trunk.lam1, trunk.lam2, trunk.lam_skip):
                assert gate.item() == pytest.approx(-1.5)
                assert gate.requires_grad
            assert trunk.pair1.dim == trunk.pair2.dim == 64
        trunk = agent.actor[0]
        xb = torch.randn(4096, 17)
        # Pair outputs match the SiTU 0.5 contribution scale; trunk matches res.
        assert trunk.pair1(trunk.in_proj(xb)).pow(2).mean().item() == pytest.approx(0.5, rel=0.2)
        assert trunk(xb).pow(2).mean().item() == pytest.approx(1.43, rel=0.2)


def test_lrelusphere_preserves_optimizer_settings_and_advnorm_flags():
    assert lrelusphere_noadvnorm.Args().learning_rate == 9.6e-3
    assert lrelusphere_noadvnorm.Args().num_minibatches == 1
    assert lrelusphere_noadvnorm.Args().norm_adv is False


def test_lrelusphere_trunks_live_on_the_unit_hypersphere():
    torch.manual_seed(0)
    agent = lrelusphere_noadvnorm.Agent(_stiglu_envs_stub())
    trunks = [m for m in (agent.actor[0], agent.critic[0]) if isinstance(m, LReluSphereTrunk)]
    assert len(trunks) == 2
    assert not any(isinstance(m, SiTUGLUBranch) for m in agent.modules())
    trunk = agent.actor[0]
    assert trunk.n_blocks == 3
    assert all(isinstance(block, LReluSqPair) for block in trunk.blocks)
    xb = torch.randn(4096, 17)
    out = trunk(xb)
    assert torch.allclose(out.norm(p=2, dim=-1), torch.ones(4096), rtol=1e-4, atol=1e-6)
    s0 = justnorm(trunk.in_proj(xb))
    assert trunk.blocks[0].lin1(s0).var().item() == pytest.approx(2.0, rel=0.15)
    assert trunk.blocks[0](s0).pow(2).mean().item() == pytest.approx(0.5, rel=0.2)


def test_sphere_1ep_splits_match_joint_losses_with_correct_flags():
    cases = [
        (sph1ep_tg, sphere_truegate_noadvnorm, True),
        (sph1ep_noadv, sphere_noadvnorm, False),
    ]
    for split_mod, joint_mod, has_truegate in cases:
        assert split_mod.Args().actor_epochs == 1
        assert split_mod.Args().critic_epochs == 10
        assert not hasattr(split_mod.Args(), "update_epochs")
        assert split_mod.Args().norm_adv is False
        assert hasattr(split_mod, "truegate_weights") == has_truegate
        torch.manual_seed(0)
        a = split_mod.Agent(_stiglu_envs_stub())
        torch.manual_seed(0)
        b = joint_mod.Agent(_stiglu_envs_stub())
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb)
        trunks = [m for m in (a.actor[0], a.critic[0]) if isinstance(m, SiTUSphereTrunk)]
        assert len(trunks) == 2
        args = split_mod.Args()
        obs = torch.randn(64, 17)
        native = torch.rand(64, 6).clamp(1e-6, 1 - 1e-6)
        adv, ret, old_v = torch.randn(64) * 3.0, torch.randn(64) * 100.0, torch.randn(64) * 100.0
        old_lp = torch.randn(64)
        if has_truegate:
            w = split_mod.truegate_weights(adv, torch.randn(64), args.dg_eta, args.dg_surprisal_clip)
            adv = adv * w
        joint, _ = joint_mod.ppo_loss(b, obs, native, old_lp, adv, ret, old_v, args)
        pa, _ = split_mod.actor_loss(a, obs, native, old_lp, adv, args)
        pv = split_mod.critic_loss(a, obs, ret, old_v, args)
        torch.testing.assert_close(pa + pv, joint, rtol=1e-4, atol=1e-5)


def test_v30_explicit_next_values_cut_trace_on_truncation():
    rewards = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[0.0, 0.0]])
    next_values = torch.tensor([[10.0, 10.0]])
    terms = torch.tensor([[0.0, 0.0]])
    truncs = torch.tensor([[0.0, 1.0]])
    advantages, returns = compute_gae_from_next_values(
        rewards, values, terms, truncs, next_values, gamma=0.99, gae_lambda=0.95
    )
    # Non-truncated: bootstrap 0.99 * 10. Truncated: v30 cuts the trace to delta
    # but still bootstraps the actual next observation (not zero).
    torch.testing.assert_close(advantages[0, 0], torch.tensor(1.0 + 0.99 * 10.0))
    torch.testing.assert_close(advantages[0, 1], torch.tensor(1.0 + 0.99 * 10.0))

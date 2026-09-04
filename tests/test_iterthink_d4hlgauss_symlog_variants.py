import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

from cleanrl.shared.hl_gauss import HLGaussSupport
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_v1 import (
    Agent as RawReturnAgent,
    Args as RawReturnArgs,
    make_env as make_rawret_env,
    value_support_bounds as rawret_value_support_bounds,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_headnorm_v1 import (
    Agent as RawReturnHeadNormAgent,
    Args as RawReturnHeadNormArgs,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_histtransfer_v1 import (
    Agent as RawReturnHistTransferAgent,
    Args as RawReturnHistTransferArgs,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_v1 import (
    Agent as RawReturnIdxTransferAgent,
    Args as RawReturnIdxTransferArgs,
    IndexedTransferBranch as RawReturnIndexedTransferBranch,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_latentmoe_v1 import (
    Agent as RawReturnIdxTransferLatentMoEAgent,
    Args as RawReturnIdxTransferLatentMoEArgs,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_latentmoe_v2 import (
    Agent as RawReturnIdxTransferLatentMoEV2Agent,
    Args as RawReturnIdxTransferLatentMoEV2Args,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_mlp10_idxtransfer_v1 import (
    Agent as RawReturnMLP10IdxTransferAgent,
    Args as RawReturnMLP10IdxTransferArgs,
    IndexedTransferLayer as RawReturnMLP10IndexedTransferLayer,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_mlp10_obsidxtransfer_v1 import (
    Agent as RawReturnMLP10ObsIdxTransferAgent,
    Args as RawReturnMLP10ObsIdxTransferArgs,
    IndexedTransferLayer as RawReturnMLP10ObsIndexedTransferLayer,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_mlp10_obsidxtransfer_justnorm_v1 import (
    Agent as RawReturnMLP10ObsIdxTransferJustNormAgent,
    Args as RawReturnMLP10ObsIdxTransferJustNormArgs,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxpreacttransfer_v1 import (
    Agent as RawReturnIdxPreActTransferAgent,
    Args as RawReturnIdxPreActTransferArgs,
    IndexedPreActivationTransferBranch as RawReturnIndexedPreActivationTransferBranch,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_ngptjustnorm_v1 import (
    Agent as RawReturnNGPTJustNormAgent,
    Args as RawReturnNGPTJustNormArgs,
    make_env as make_ngptjustnorm_env,
    value_support_bounds as ngptjustnorm_value_support_bounds,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_ngptjustnorm_v2 import (
    Agent as RawReturnNGPTResidualAgent,
    Args as RawReturnNGPTResidualArgs,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_ngptjustnorm_v3 import (
    Agent as RawReturnNGPTResidualV3Agent,
    Args as RawReturnNGPTResidualV3Args,
    make_env as make_ngptresidual_v3_env,
    value_support_bounds as ngptresidual_v3_value_support_bounds,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_ngptjustnorm_v4 import (
    Agent as RawReturnNGPTResidualV4Agent,
    Args as RawReturnNGPTResidualV4Args,
)
from cleanrl.iterthink.v24_d4hlgauss.rawret.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_ngptjustnorm_v5 import (
    Agent as RawReturnNGPTResidualV5Agent,
    Args as RawReturnNGPTResidualV5Args,
)
from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_symlogobs_v1 import (
    Args as SymlogObsArgs,
    make_env as make_symlogobs_env,
    symlog_obs,
)
from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_tokenobs_v2 import (
    Args as TokenObsArgs,
    make_env as make_tokenobs_env,
    ObsTokenStem,
)
def wrapper_names(env):
    names = []
    while isinstance(env, gym.Wrapper):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_raw_return_variant_disables_reward_normalization_and_uses_v149_critic():
    args = RawReturnArgs()

    assert args.normalize_reward is False
    assert args.clip_reward is False
    assert args.vf_coef == 1.0
    assert args.value_sigma_to_bin_ratio == 0.5
    assert not hasattr(args, "critic_mtp_horizon")

    support_min, support_max = rawret_value_support_bounds(args)
    assert np.allclose([support_min, support_max], [args.v_min, args.v_max])
    assert np.allclose([support_min, support_max], [-9.90353755128617, 9.90353755128617])
    support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    zero_logits = torch.zeros(1, args.num_bins)
    assert support.support_is_edges is True
    assert support.to_expected_scalar(zero_logits).abs().item() < 1e-3

    env = make_rawret_env("Pendulum-v1", 0, False, "test_rawret", args.gamma, args.normalize_reward, args.clip_reward)()
    try:
        names = wrapper_names(env)
        assert "NormalizeObservation" in names
        assert "NormalizeReward" not in names
        assert "TransformReward" not in names
    finally:
        env.close()

    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    agent = RawReturnAgent(DummyVecEnv(), RawReturnArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31))
    logits = agent.get_value(torch.zeros(3, 4))
    assert agent.critic_head.bias is None
    assert torch.allclose(agent.critic_head.weight, torch.zeros_like(agent.critic_head.weight))
    assert logits.shape == (3, 31)
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_raw_return_headnorm_variant_uses_biasless_pre_normed_actor_and_critic_heads():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    beta_agent = RawReturnHeadNormAgent(
        DummyVecEnv(),
        RawReturnHeadNormArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31),
    )
    gaussian_agent = RawReturnHeadNormAgent(
        DummyVecEnv(),
        RawReturnHeadNormArgs(hidden=16, k_blocks=1, n_experts=1, num_bins=31, actor_dist="gaussian"),
    )

    for agent in (beta_agent, gaussian_agent):
        assert agent.critic_head.bias is None
        assert isinstance(agent.actor_head_norm, torch.nn.RMSNorm)
        assert isinstance(agent.critic_head_norm, torch.nn.RMSNorm)
        assert agent.actor_head_norm.elementwise_affine is False
        assert agent.critic_head_norm.elementwise_affine is False
        assert torch.allclose(agent.critic_head.weight, torch.zeros_like(agent.critic_head.weight))

    assert beta_agent.actor_alpha_head.bias is None
    assert beta_agent.actor_beta_head.bias is None
    assert gaussian_agent.actor_head.bias is None
    assert gaussian_agent.actor_logvar_head.bias is None

    obs = torch.zeros(3, 4)
    logits = beta_agent.get_value(obs)
    action, z, log_prob, entropy, value_logits = beta_agent.get_action_and_value(obs)
    assert logits.shape == (3, 31)
    assert value_logits.shape == (3, 31)
    assert action.shape == (3, 2)
    assert z.shape == (3, 2)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)


def test_raw_return_histtransfer_variant_feeds_history_into_branch_transfer():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    torch.manual_seed(0)
    agent = RawReturnHistTransferAgent(
        DummyVecEnv(),
        RawReturnHistTransferArgs(hidden=16, k_blocks=2, n_experts=2, num_bins=31),
    )
    first_block, second_block = agent.trunk.blocks

    assert first_block.dense.in_linear.in_features == 32
    assert first_block.experts[0].in_linear.in_features == 32
    assert second_block.dense.in_linear.in_features == 48
    assert second_block.experts[0].in_linear.in_features == 48
    assert second_block.in_proj.in_features == 32

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat.pow(2).sum()
    loss.backward()

    first_history_grad = first_block.dense.in_linear.weight.grad[:, 16:]
    second_history_grad = second_block.experts[0].in_linear.weight.grad[:, 16:]
    assert first_history_grad.norm().item() > 0.0
    assert second_history_grad.norm().item() > 0.0


def test_raw_return_idxtransfer_variant_uses_same_index_history_only():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    torch.manual_seed(0)
    agent = RawReturnIdxTransferAgent(
        DummyVecEnv(),
        RawReturnIdxTransferArgs(hidden=16, k_blocks=3, n_experts=2, num_bins=31),
    )
    first_block, second_block, third_block = agent.trunk.blocks

    assert first_block.dense.current_linear.in_features == 16
    assert second_block.dense.current_linear.in_features == 16
    assert third_block.dense.current_linear.in_features == 16
    assert first_block.dense.history_weight.shape == (1, 16)
    assert second_block.dense.history_weight.shape == (2, 16)
    assert third_block.dense.history_weight.shape == (3, 16)
    assert third_block.experts[0].history_weight.shape == (3, 16)
    assert third_block.in_proj.in_features == 48

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat.pow(2).sum()
    loss.backward()

    assert first_block.dense.history_weight.grad.norm().item() > 0.0
    assert second_block.experts[0].history_weight.grad.norm().item() > 0.0
    assert third_block.dense.history_weight.grad.norm().item() > 0.0


def test_raw_return_idxtransfer_latentmoe_variant_uses_latent_topk_experts():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    default_agent = RawReturnIdxTransferLatentMoEAgent(
        DummyVecEnv(),
        RawReturnIdxTransferLatentMoEArgs(hidden=16, k_blocks=1, n_experts=4, num_bins=31),
    )
    default_block = default_agent.trunk.blocks[0]
    assert default_block.moe.router.out_features == 16
    assert default_block.moe.top_k == 8

    torch.manual_seed(0)
    sparse_agent = RawReturnIdxTransferLatentMoEAgent(
        DummyVecEnv(),
        RawReturnIdxTransferLatentMoEArgs(
            hidden=16,
            k_blocks=3,
            n_experts=2,
            num_bins=31,
            latent_moe_compression=4,
            latent_moe_base_top_k=1,
        ),
    )
    first_block, _, third_block = sparse_agent.trunk.blocks

    assert first_block.moe.router.in_features == 16
    assert first_block.moe.router.out_features == 8
    assert first_block.moe.top_k == 4
    assert first_block.moe.latent_dim == 4
    assert first_block.moe.down.out_features == 4
    assert first_block.moe.up.in_features == 4
    assert first_block.moe.expert_current_weight.shape == (8, 16, 4)
    assert first_block.moe.expert_out_weight.shape == (8, 4, 16)
    assert third_block.moe.expert_history_weight.shape == (8, 3, 16)
    assert third_block.dense.history_weight.shape == (3, 16)

    _, critic_feat = sparse_agent._trunks(torch.randn(5, 4))
    loss = critic_feat.pow(2).sum()
    loss.backward()

    assert first_block.moe.router.weight.grad.norm().item() > 0.0
    assert first_block.moe.down.weight.grad.norm().item() > 0.0
    assert first_block.moe.up.weight.grad.norm().item() > 0.0
    assert first_block.moe.expert_current_weight.grad.norm().item() > 0.0
    assert first_block.moe.expert_out_weight.grad.norm().item() > 0.0
    assert first_block.moe.expert_history_weight.grad.norm().item() > 0.0


def test_raw_return_idxtransfer_latentmoe_v2_uses_wider_intermediate_and_topk16():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    torch.manual_seed(0)
    agent = RawReturnIdxTransferLatentMoEV2Agent(
        DummyVecEnv(),
        RawReturnIdxTransferLatentMoEV2Args(hidden=64, k_blocks=1, n_experts=16, num_bins=31),
    )
    block = agent.trunk.blocks[0]

    assert block.moe.latent_dim == 16
    assert block.moe.intermediate_dim == 128
    assert block.moe.router.out_features == 64
    assert block.moe.top_k == 16
    assert block.moe.expert_current_weight.shape == (64, 128, 16)
    assert block.moe.expert_out_weight.shape == (64, 16, 128)
    assert block.moe.expert_history_weight.shape == (64, 1, 128)

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat.pow(2).sum()
    loss.backward()

    assert block.moe.router.weight.grad.norm().item() > 0.0
    assert block.moe.expert_current_weight.grad.norm().item() > 0.0
    assert block.moe.expert_out_weight.grad.norm().item() > 0.0


def test_indexed_transfer_branch_is_invariant_to_off_index_history():
    branch = RawReturnIndexedTransferBranch(H=4, history_dim=12)
    with torch.no_grad():
        branch.current_linear.weight.zero_()
        branch.current_linear.bias.zero_()
        branch.out_linear.weight.zero_()
        branch.out_linear.weight.copy_(torch.eye(4))
        branch.out_linear.bias.zero_()
        branch.history_weight.zero_()
        branch.history_weight[:, 0] = 1.0

    x = torch.zeros(2, 4)
    history = torch.zeros(2, 12)
    history.view(2, 3, 4)[:, :, 0] = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    base = branch(x, history)

    off_index_perturbed = history.clone()
    off_index_perturbed.view(2, 3, 4)[:, :, 1:] = torch.randn(2, 3, 3) * 100.0
    same_index_perturbed = history.clone()
    same_index_perturbed.view(2, 3, 4)[:, 1, 0] += 1.0

    assert torch.allclose(branch(x, off_index_perturbed), base)
    assert not torch.allclose(branch(x, same_index_perturbed), base)


def test_raw_return_mlp10_idxtransfer_variant_removes_dense_residual_blocks():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    args = RawReturnMLP10IdxTransferArgs(num_bins=31)
    assert args.hidden == 128
    assert args.mlp_depth == 10
    assert not hasattr(args, "k_blocks")
    assert not hasattr(args, "n_experts")

    agent = RawReturnMLP10IdxTransferAgent(
        DummyVecEnv(),
        RawReturnMLP10IdxTransferArgs(hidden=128, mlp_depth=10, num_bins=31),
    )
    trunk = agent.trunk

    assert not hasattr(trunk, "blocks")
    assert len(trunk.layers) == 10
    assert trunk.layers[0].current_linear.in_features == 17
    assert trunk.layers[1].current_linear.in_features == 128
    assert trunk.layers[0].history_weight is None
    assert trunk.layers[1].history_weight is None
    assert trunk.layers[2].history_weight.shape == (1, 128)
    assert trunk.layers[9].history_weight.shape == (8, 128)

    obs = torch.randn(32, 17)
    action, z, log_prob, entropy, value_logits = agent.get_action_and_value(obs)
    for tensor in (action, z, log_prob, entropy, value_logits):
        assert torch.isfinite(tensor).all()

    _, critic_feat = agent._trunks(obs)
    assert torch.isfinite(critic_feat).all()
    loss = critic_feat.pow(2).sum()
    loss.backward()

    assert torch.isfinite(trunk.layers[2].history_weight.grad).all()
    assert torch.isfinite(trunk.layers[9].history_weight.grad).all()
    assert trunk.layers[2].history_weight.grad.norm().item() > 0.0
    assert trunk.layers[9].history_weight.grad.norm().item() > 0.0


def test_mlp10_indexed_transfer_layer_is_invariant_to_off_index_history():
    layer = RawReturnMLP10IndexedTransferLayer(in_dim=4, H=4, history_slots=3)
    with torch.no_grad():
        layer.current_linear.weight.zero_()
        layer.current_linear.bias.zero_()
        layer.current_linear.bias[1] = 1.0
        layer.history_weight.zero_()
        layer.history_weight[:, 0] = 1.0

    x = torch.zeros(2, 4)
    history_feats = [torch.zeros(2, 4) for _ in range(3)]
    for slot, feat in enumerate(history_feats):
        feat[:, 0] = torch.tensor([slot + 1.0, slot + 0.5])
    base = layer(x, history_feats)

    off_index_perturbed = [feat.clone() for feat in history_feats]
    for feat in off_index_perturbed:
        feat[:, 2:] = torch.randn(2, 2) * 100.0
    same_index_perturbed = [feat.clone() for feat in history_feats]
    same_index_perturbed[1][:, 0] += 1.0

    assert torch.allclose(layer(x, off_index_perturbed), base)
    assert not torch.allclose(layer(x, same_index_perturbed), base)


def test_raw_return_mlp10_obsidxtransfer_variant_uses_obs_stem_transfer():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    args = RawReturnMLP10ObsIdxTransferArgs(num_bins=31)
    assert args.hidden == 128
    assert args.mlp_depth == 10
    assert not hasattr(args, "k_blocks")
    assert not hasattr(args, "n_experts")

    agent = RawReturnMLP10ObsIdxTransferAgent(
        DummyVecEnv(),
        RawReturnMLP10ObsIdxTransferArgs(hidden=128, mlp_depth=10, num_bins=31),
    )
    trunk = agent.trunk

    assert not hasattr(trunk, "blocks")
    assert trunk.stem.current_linear.in_features == 17
    assert trunk.stem.history_weight is None
    assert len(trunk.layers) == 10
    assert trunk.layers[0].current_linear.in_features == 128
    assert trunk.layers[0].history_weight is None
    assert trunk.layers[1].history_weight.shape == (1, 128)
    assert trunk.layers[9].history_weight.shape == (9, 128)

    obs = torch.randn(32, 17)
    action, z, log_prob, entropy, value_logits = agent.get_action_and_value(obs)
    for tensor in (action, z, log_prob, entropy, value_logits):
        assert torch.isfinite(tensor).all()

    _, critic_feat = agent._trunks(obs)
    assert torch.isfinite(critic_feat).all()
    loss = critic_feat.pow(2).sum()
    loss.backward()

    assert torch.isfinite(trunk.layers[1].history_weight.grad).all()
    assert torch.isfinite(trunk.layers[9].history_weight.grad).all()
    assert trunk.layers[1].history_weight.grad.norm().item() > 0.0
    assert trunk.layers[9].history_weight.grad.norm().item() > 0.0


def test_mlp10_obs_indexed_transfer_layer_is_invariant_to_off_index_history():
    layer = RawReturnMLP10ObsIndexedTransferLayer(in_dim=4, H=4, history_slots=3)
    with torch.no_grad():
        layer.current_linear.weight.zero_()
        layer.current_linear.bias.zero_()
        layer.current_linear.bias[1] = 1.0
        layer.history_weight.zero_()
        layer.history_weight[:, 0] = 1.0

    x = torch.zeros(2, 4)
    history_feats = [torch.zeros(2, 4) for _ in range(3)]
    for slot, feat in enumerate(history_feats):
        feat[:, 0] = torch.tensor([slot + 1.0, slot + 0.5])
    base = layer(x, history_feats)

    off_index_perturbed = [feat.clone() for feat in history_feats]
    for feat in off_index_perturbed:
        feat[:, 2:] = torch.randn(2, 2) * 100.0
    same_index_perturbed = [feat.clone() for feat in history_feats]
    same_index_perturbed[1][:, 0] += 1.0

    assert torch.allclose(layer(x, off_index_perturbed), base)
    assert not torch.allclose(layer(x, same_index_perturbed), base)


def test_raw_return_mlp10_obsidxtransfer_justnorm_variant_normalizes_trunk_weights():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    agent = RawReturnMLP10ObsIdxTransferJustNormAgent(
        DummyVecEnv(),
        RawReturnMLP10ObsIdxTransferJustNormArgs(hidden=128, mlp_depth=10, num_bins=31),
    )
    trunk = agent.trunk

    assert trunk.layers[9].history_weight.shape == (9, 128)
    assert torch.allclose(trunk.stem.current_linear.weight.norm(dim=1), torch.ones(128), atol=1e-6)
    assert torch.allclose(trunk.layers[0].current_linear.weight.norm(dim=1), torch.ones(128), atol=1e-6)
    assert torch.allclose(trunk.layers[9].current_linear.weight.norm(dim=1), torch.ones(128), atol=1e-6)

    actor_head_before = agent.actor_alpha_head.weight.detach().clone()
    critic_head_before = agent.critic_head.weight.detach().clone()
    transfer_before = trunk.layers[9].history_weight.detach().clone()
    with torch.no_grad():
        trunk.stem.current_linear.weight.mul_(3.0)
        trunk.layers[9].current_linear.weight.mul_(4.0)
        trunk.layers[9].history_weight.mul_(5.0)
    agent.normalize_ngpt_weights()

    assert torch.allclose(trunk.stem.current_linear.weight.norm(dim=1), torch.ones(128), atol=1e-6)
    assert torch.allclose(trunk.layers[9].current_linear.weight.norm(dim=1), torch.ones(128), atol=1e-6)
    assert torch.allclose(trunk.layers[9].history_weight, transfer_before * 5.0)
    assert torch.allclose(agent.actor_alpha_head.weight, actor_head_before)
    assert torch.allclose(agent.critic_head.weight, critic_head_before)

    obs = torch.randn(32, 17)
    action, z, log_prob, entropy, value_logits = agent.get_action_and_value(obs)
    for tensor in (action, z, log_prob, entropy, value_logits):
        assert torch.isfinite(tensor).all()

    _, critic_feat = agent._trunks(obs)
    loss = critic_feat.pow(2).sum()
    loss.backward()
    assert torch.isfinite(trunk.layers[9].history_weight.grad).all()
    assert trunk.layers[9].history_weight.grad.norm().item() > 0.0


def test_raw_return_mlp10_obsidxtransfer_justnorm_reprojects_after_optimizer_steps():
    source = Path(
        "cleanrl/iterthink/v24_d4hlgauss/rawret/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_mlp10_obsidxtransfer_justnorm_v1.py"
    ).read_text()
    assert source.count("optimizer.step()\n                    agent.normalize_ngpt_weights()") == 2


def test_raw_return_idxpreacttransfer_variant_uses_activated_same_index_history():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    torch.manual_seed(0)
    agent = RawReturnIdxPreActTransferAgent(
        DummyVecEnv(),
        RawReturnIdxPreActTransferArgs(hidden=16, k_blocks=3, n_experts=2, num_bins=31),
    )
    first_block, second_block, third_block = agent.trunk.blocks

    assert first_block.dense.current_linear.in_features == 16
    assert second_block.dense.current_linear.in_features == 16
    assert third_block.dense.current_linear.in_features == 16
    assert first_block.dense.history_weight.shape == (1, 16)
    assert second_block.dense.history_weight.shape == (2, 16)
    assert third_block.dense.history_weight.shape == (3, 16)
    assert third_block.experts[0].history_weight.shape == (3, 16)
    assert third_block.in_proj.in_features == 48

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat.pow(2).sum()
    loss.backward()

    assert first_block.dense.history_weight.grad.norm().item() > 0.0
    assert second_block.experts[0].history_weight.grad.norm().item() > 0.0
    assert third_block.dense.history_weight.grad.norm().item() > 0.0


def test_indexed_preactivation_transfer_branch_activates_history_before_sum():
    branch = RawReturnIndexedPreActivationTransferBranch(H=4, history_dim=8)
    with torch.no_grad():
        branch.current_linear.weight.zero_()
        branch.current_linear.bias.copy_(torch.tensor([-3.0, 2.0, 0.0, 0.0]))
        branch.out_linear.weight.zero_()
        branch.out_linear.weight.copy_(torch.eye(4))
        branch.out_linear.bias.zero_()
        branch.history_weight.zero_()
        branch.history_weight[:, 0] = 1.0

    x = torch.zeros(1, 4)
    history = torch.zeros(1, 8)
    history.view(1, 2, 4)[:, :, 0] = torch.tensor([[-2.0, 3.0]])
    history.view(1, 2, 4)[:, :, 1:] = torch.randn(1, 2, 3) * 100.0

    out = branch(x, history)

    expected = torch.zeros_like(out)
    expected[:, 0] = 9.0
    expected[:, 1] = 4.0
    assert torch.allclose(out, expected)


def test_indexed_preactivation_transfer_branch_is_invariant_to_off_index_history():
    branch = RawReturnIndexedPreActivationTransferBranch(H=4, history_dim=12)
    with torch.no_grad():
        branch.current_linear.weight.zero_()
        branch.current_linear.bias.zero_()
        branch.out_linear.weight.zero_()
        branch.out_linear.weight.copy_(torch.eye(4))
        branch.out_linear.bias.zero_()
        branch.history_weight.zero_()
        branch.history_weight[:, 0] = torch.tensor([1.0, -0.5, 0.25])
        branch.history_weight[:, 2] = torch.tensor([-0.25, 0.75, 1.0])

    x = torch.zeros(2, 4)
    history = torch.zeros(2, 12)
    history.view(2, 3, 4)[:, :, 0] = torch.tensor([[1.0, -2.0, 3.0], [0.5, 1.5, -2.5]])
    history.view(2, 3, 4)[:, :, 2] = torch.tensor([[-1.0, 2.0, 0.5], [3.0, -0.5, 1.0]])
    base = branch(x, history)

    off_index_perturbed = history.clone()
    off_index_perturbed.view(2, 3, 4)[:, :, [1, 3]] = torch.randn(2, 3, 2) * 100.0
    same_index_perturbed = history.clone()
    same_index_perturbed.view(2, 3, 4)[:, 0, 2] += 1.0

    assert torch.allclose(branch(x, off_index_perturbed), base)
    assert not torch.allclose(branch(x, same_index_perturbed), base)


def test_raw_return_ngptjustnorm_variant_preserves_rawret_defaults_and_excludes_heads():
    args = RawReturnNGPTJustNormArgs()

    assert args.normalize_reward is False
    assert args.clip_reward is False
    assert args.vf_coef == 1.0
    assert args.value_sigma_to_bin_ratio == 0.5
    assert not hasattr(args, "critic_mtp_horizon")
    assert np.allclose(ngptjustnorm_value_support_bounds(args), (args.v_min, args.v_max))

    env = make_ngptjustnorm_env(
        "Pendulum-v1",
        0,
        False,
        "test_ngptjustnorm",
        args.gamma,
        args.normalize_reward,
        args.clip_reward,
    )()
    try:
        names = wrapper_names(env)
        assert "NormalizeObservation" in names
        assert "TransformObservation" in names
        assert "NormalizeReward" not in names
        assert "TransformReward" not in names
    finally:
        env.close()

    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    agent = RawReturnNGPTJustNormAgent(
        DummyVecEnv(),
        RawReturnNGPTJustNormArgs(hidden=16, k_blocks=1, n_experts=2, num_bins=31),
    )
    block = agent.trunk.blocks[0]

    assert torch.allclose(agent.trunk.entry.weight.norm(dim=1), torch.ones(16), atol=1e-6)
    assert torch.allclose(block.in_proj.weight.norm(dim=1), torch.ones(16), atol=1e-6)
    assert torch.allclose(block.dense[0].weight.norm(dim=1), torch.ones(16), atol=1e-6)
    assert torch.allclose(block.dense[2].weight.norm(dim=0), torch.ones(16), atol=1e-6)
    assert torch.allclose(agent.trunk.out_proj.weight.norm(dim=0), torch.ones(32), atol=1e-6)
    assert torch.allclose(agent.critic_head.weight, torch.zeros_like(agent.critic_head.weight))
    assert agent.actor_alpha_head.weight.norm(dim=1).max().item() < 0.02
    assert agent.actor_beta_head.weight.norm(dim=1).max().item() < 0.02

    actor_head_before = agent.actor_alpha_head.weight.detach().clone()
    critic_head_before = agent.critic_head.weight.detach().clone()
    with torch.no_grad():
        agent.trunk.entry.weight.mul_(3.0)
    agent.normalize_ngpt_weights()
    assert torch.allclose(agent.trunk.entry.weight.norm(dim=1), torch.ones(16), atol=1e-6)
    assert torch.allclose(agent.actor_alpha_head.weight, actor_head_before)
    assert torch.allclose(agent.critic_head.weight, critic_head_before)


def test_raw_return_ngpt_residual_variant_excludes_router_and_uses_max_strength_residuals():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    agent = RawReturnNGPTResidualAgent(
        DummyVecEnv(),
        RawReturnNGPTResidualArgs(hidden=16, k_blocks=1, n_experts=2, num_bins=31),
    )
    block = agent.trunk.blocks[0]

    assert block.gate not in block.ngpt_input_linears
    assert torch.allclose(block.input_alpha, torch.ones_like(block.input_alpha))
    assert torch.allclose(block.dense_alpha, torch.ones_like(block.dense_alpha))
    assert torch.allclose(block.moe_alpha, torch.ones_like(block.moe_alpha))

    gate_before = block.gate.weight.detach().clone()
    with torch.no_grad():
        block.gate.weight.mul_(3.0)
        agent.trunk.entry.weight.mul_(3.0)
    agent.normalize_ngpt_weights()
    assert torch.allclose(block.gate.weight, gate_before * 3.0)
    assert torch.allclose(agent.trunk.entry.weight.norm(dim=1), torch.ones(16), atol=1e-6)

    actor_feat, critic_feat = agent._trunks(torch.randn(3, 4))
    assert torch.allclose(actor_feat.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(critic_feat.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_raw_return_ngpt_residual_v3_combines_branches_without_dead_dense_gradient():
    args = RawReturnNGPTResidualV3Args()

    assert args.normalize_reward is False
    assert args.clip_reward is False
    assert args.vf_coef == 1.0
    assert args.value_sigma_to_bin_ratio == 0.5
    assert not hasattr(args, "critic_mtp_horizon")
    support_min, support_max = ngptresidual_v3_value_support_bounds(args)
    assert np.allclose([support_min, support_max], [args.v_min, args.v_max])
    support = HLGaussSupport(
        args.num_bins,
        support_min,
        support_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
        use_symlog=args.value_symlog,
        support_is_edges=True,
    )
    assert support.support_is_edges is True
    assert support.to_expected_scalar(torch.zeros(1, args.num_bins)).abs().item() < 1e-3

    env = make_ngptresidual_v3_env(
        "Pendulum-v1",
        0,
        False,
        "test_ngptresidual_v3",
        args.gamma,
        args.normalize_reward,
        args.clip_reward,
    )()
    try:
        names = wrapper_names(env)
        assert "NormalizeObservation" in names
        assert "TransformObservation" in names
        assert "NormalizeReward" not in names
        assert "TransformReward" not in names
    finally:
        env.close()

    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    agent = RawReturnNGPTResidualV3Agent(
        DummyVecEnv(),
        RawReturnNGPTResidualV3Args(hidden=16, k_blocks=1, n_experts=2, num_bins=31),
    )
    block = agent.trunk.blocks[0]

    assert block.gate not in block.ngpt_input_linears
    assert torch.allclose(block.input_alpha, torch.ones_like(block.input_alpha))
    assert torch.allclose(block.branch_alpha, torch.ones_like(block.branch_alpha))

    actor_feat, critic_feat = agent._trunks(torch.randn(3, 4))
    assert torch.allclose(actor_feat.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(critic_feat.norm(dim=-1), torch.ones(3), atol=1e-5)

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat[:, 0].sum()
    loss.backward()
    assert block.dense[0].weight.grad is not None
    assert block.dense[2].weight.grad is not None
    assert block.experts[0][0].weight.grad is not None
    assert block.dense[0].weight.grad.norm().item() > 0.0
    assert block.dense[2].weight.grad.norm().item() > 0.0
    assert block.experts[0][0].weight.grad.norm().item() > 0.0


def test_raw_return_ngpt_residual_v4_uses_reference_residual_scale():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    agent = RawReturnNGPTResidualV4Agent(
        DummyVecEnv(),
        RawReturnNGPTResidualV4Args(hidden=16, k_blocks=1, n_experts=2, num_bins=31),
    )
    block = agent.trunk.blocks[0]

    assert block.gate not in block.ngpt_input_linears
    assert torch.allclose(block.input_alpha, torch.full_like(block.input_alpha, 0.05))
    assert torch.allclose(block.branch_alpha, torch.full_like(block.branch_alpha, 0.05))

    actor_feat, critic_feat = agent._trunks(torch.randn(3, 4))
    assert torch.allclose(actor_feat.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(critic_feat.norm(dim=-1), torch.ones(3), atol=1e-5)

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat[:, 0].sum()
    loss.backward()
    assert block.dense[0].weight.grad is not None
    assert block.experts[0][0].weight.grad is not None
    assert block.dense[0].weight.grad.norm().item() > 0.0
    assert block.experts[0][0].weight.grad.norm().item() > 0.0


def test_raw_return_ngpt_residual_v5_uses_reference_alpha_parameterization():
    class DummyVecEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    agent = RawReturnNGPTResidualV5Agent(
        DummyVecEnv(),
        RawReturnNGPTResidualV5Args(hidden=16, k_blocks=1, n_experts=2, num_bins=31),
    )
    block = agent.trunk.blocks[0]

    expected_stored = torch.full_like(block.input_alpha, 16 ** -0.5)
    expected_effective = torch.full_like(block.input_alpha, 0.05)
    input_effective = (block.input_alpha * (block.alpha_init_value / block.alpha_init_scaling)).abs()
    branch_effective = (block.branch_alpha * (block.alpha_init_value / block.alpha_init_scaling)).abs()

    assert block.gate not in block.ngpt_input_linears
    assert torch.allclose(block.input_alpha, expected_stored)
    assert torch.allclose(block.branch_alpha, expected_stored)
    assert torch.allclose(input_effective, expected_effective)
    assert torch.allclose(branch_effective, expected_effective)

    actor_feat, critic_feat = agent._trunks(torch.randn(3, 4))
    assert torch.allclose(actor_feat.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(critic_feat.norm(dim=-1), torch.ones(3), atol=1e-5)

    _, critic_feat = agent._trunks(torch.randn(5, 4))
    loss = critic_feat[:, 0].sum()
    loss.backward()
    assert block.dense[0].weight.grad is not None
    assert block.dense[2].weight.grad is not None
    assert block.experts[0][0].weight.grad is not None
    assert block.experts[0][2].weight.grad is not None
    assert block.dense[0].weight.grad.norm().item() > 0.0
    assert block.dense[2].weight.grad.norm().item() > 0.0
    assert block.experts[0][0].weight.grad.norm().item() > 0.0
    assert block.experts[0][2].weight.grad.norm().item() > 0.0


def test_symlog_observation_variant_replaces_observation_normalization_only():
    args = SymlogObsArgs()
    obs = np.array([-22025.465794806718, -1.0, 0.0, 1.0, 22025.465794806718], dtype=np.float64)

    assert np.allclose(symlog_obs(obs), [-10.0, -np.log(2.0), 0.0, np.log(2.0), 10.0])

    env = make_symlogobs_env("Pendulum-v1", 0, False, "test_symlogobs", args.gamma)()
    try:
        names = wrapper_names(env)
        assert "NormalizeObservation" not in names
        assert "TransformObservation" in names
        assert "NormalizeReward" in names
        assert "TransformReward" in names
    finally:
        env.close()


def test_tokenized_raw_observation_stem_uses_nonshared_per_obs_projection_and_flat_projection():
    args = TokenObsArgs()
    env = make_tokenobs_env("Pendulum-v1", 0, False, "test_tokenobs", args.gamma)()
    try:
        names = wrapper_names(env)
        assert "NormalizeObservation" not in names
        assert "TransformObservation" not in names
        assert "NormalizeReward" in names
        assert "TransformReward" in names
    finally:
        env.close()

    obs_dim = 7
    token_dim = 16
    out_dim = 8
    stem = ObsTokenStem(obs_dim, token_dim, out_dim)
    obs = torch.linspace(-4.0, 4.0, steps=2 * obs_dim).reshape(2, obs_dim)

    tokens = stem.encode_tokens(obs)
    out = stem(obs)

    assert tokens.shape == (2, obs_dim, token_dim)
    assert out.shape == (2, out_dim)
    assert stem.obs_weight.shape == (obs_dim, token_dim)
    assert stem.obs_bias.shape == (obs_dim, token_dim)
    assert stem.token_norm.elementwise_affine is False
    assert stem.out_proj.in_features == obs_dim * token_dim
    assert not hasattr(stem, "mean_pool")
    assert not hasattr(stem, "readout_queries")
    assert not hasattr(stem, "scalar_proj")
    assert not hasattr(stem, "token_proj")
    assert torch.allclose(tokens.pow(2).mean(dim=-1), torch.ones(2, obs_dim), atol=1e-3)

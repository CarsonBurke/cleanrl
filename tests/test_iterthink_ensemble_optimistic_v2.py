import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_v1 import Args as D4Args
from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_ensemble_optimistic_v2 import (
    Agent,
    Args,
    ThinkTrunk,
    decode_value,
    optimistic_policy_gae,
    rank_corr,
)


def _standard_reward_gae(rewards, dones, next_done, values, next_value, gamma, lam):
    """Reference standard reward GAE (the base recursion) for kappa==0 parity."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    return advantages


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def _agent(**overrides):
    kw = dict(hidden=16, k_blocks=1, n_experts=2, num_bins=31)
    kw.update(overrides)
    args = Args(**kw)
    return Agent(DummyVecEnv(), args), args


# 1. decode_value -----------------------------------------------------------
def test_decode_value_identical_heads_zero_epistemic():
    support = torch.linspace(-10.0, 10.0, 31)
    single = torch.randn(5, 31)              # (B, n)
    logits = single.unsqueeze(0).repeat(4, 1, 1)  # K identical heads
    v_mean, v_epi = decode_value(logits, support)

    probs = torch.softmax(single, dim=-1)
    v_single = (probs * support).sum(dim=-1)
    assert torch.allclose(v_epi, torch.zeros(5), atol=1e-6)
    assert torch.allclose(v_mean, v_single, atol=1e-6)


def test_decode_value_differing_heads_population_std():
    support = torch.linspace(-10.0, 10.0, 31)
    logits = torch.randn(4, 5, 31)
    v_mean, v_epi = decode_value(logits, support)

    probs = torch.softmax(logits, dim=-1)
    v_k = (probs * support).sum(dim=-1)      # (K, B)
    assert (v_epi > 0).all()
    assert torch.allclose(v_mean, v_k.mean(dim=0), atol=1e-6)
    assert torch.allclose(v_epi, v_k.std(dim=0, unbiased=False), atol=1e-6)


# 2. optimistic_policy_gae kappa==0 parity ----------------------------------
def test_optimistic_gae_kappa0_equals_standard_gae():
    torch.manual_seed(0)
    T, B = 7, 3
    rewards = torch.randn(T, B)
    dones = (torch.rand(T, B) > 0.8).float()
    next_done = (torch.rand(B) > 0.8).float()
    values = torch.randn(T, B)
    values_epi = torch.rand(T, B)
    boot_epi = torch.rand(B)
    next_value = torch.randn(1, B)
    gamma, lam = 0.99, 0.95

    ref = _standard_reward_gae(rewards, dones, next_done, values, next_value, gamma, lam)
    got = optimistic_policy_gae(
        rewards, dones, next_done, values, values_epi, boot_epi, next_value,
        kappa=0.0, gamma=gamma, lam=lam,
    )
    assert torch.allclose(got, ref, atol=1e-6)


# 3. optimistic_policy_gae kappa>0 directional effect -----------------------
def test_optimistic_gae_high_epi_next_state_raises_preceding_advantage():
    # Deterministic single env, no terminations. Inject epistemic mass ONLY at t=2;
    # the bonus on step t-1's bootstrap (= kappa*v_epi[t]) raises advantage at t=1.
    T, B = 4, 1
    rewards = torch.zeros(T, B)
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)
    values = torch.zeros(T, B)
    next_value = torch.zeros(1, B)
    gamma, lam = 0.99, 0.95

    values_epi = torch.zeros(T, B)
    values_epi[2, 0] = 1.0                    # high disagreement at s_2
    boot_epi = torch.zeros(B)

    base = optimistic_policy_gae(
        rewards, dones, next_done, values, torch.zeros(T, B), torch.zeros(B),
        next_value, kappa=0.0, gamma=gamma, lam=lam,
    )
    opt = optimistic_policy_gae(
        rewards, dones, next_done, values, values_epi, boot_epi,
        next_value, kappa=2.0, gamma=gamma, lam=lam,
    )
    assert opt[1, 0] > base[1, 0] + 1e-6      # state preceding high-epi next-state
    assert opt[0, 0] > base[0, 0] + 1e-6      # earlier state via GAE backprop
    assert torch.allclose(opt[2, 0], base[2, 0], atol=1e-6)   # s_2 itself unaffected
    assert torch.allclose(opt[3, 0], base[3, 0], atol=1e-6)
    # Exact magnitude at t=1: extra delta is gamma*kappa*v_epi[2] = 0.99*2*1.0.
    assert torch.allclose(opt[1, 0] - base[1, 0], torch.tensor(gamma * 2.0 * 1.0), atol=1e-5)


# 4. terminal masking -------------------------------------------------------
def test_optimistic_gae_terminal_zeroes_bonus_and_value():
    # A done at t+1 must zero BOTH the bootstrap value and the optimism bonus at t.
    T, B = 3, 1
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[5.0], [5.0], [5.0]])
    values_epi = torch.tensor([[2.0], [2.0], [2.0]])
    boot_epi = torch.tensor([3.0])
    next_value = torch.tensor([[5.0]])
    next_done = torch.zeros(B)
    gamma, lam = 0.99, 0.95

    dones = torch.zeros(T, B)
    dones[1, 0] = 1.0                          # s_1 is terminal => nonterminal at t=0 is 0

    adv = optimistic_policy_gae(
        rewards, dones, next_done, values, values_epi, boot_epi, next_value,
        kappa=10.0, gamma=gamma, lam=lam,
    )
    assert torch.allclose(adv[0, 0], rewards[0, 0] - values[0, 0], atol=1e-6)
    adv_noterm = optimistic_policy_gae(
        rewards, torch.zeros(T, B), next_done, values, values_epi, boot_epi, next_value,
        kappa=10.0, gamma=gamma, lam=lam,
    )
    assert not torch.allclose(adv_noterm[0, 0], rewards[0, 0] - values[0, 0], atol=1e-6)


# 5. bootstrap mask value loss ----------------------------------------------
def _masked_value_loss(value_logits, target_probs, mask):
    value_log_probs = torch.log_softmax(value_logits, dim=-1)        # (K, mb, n)
    ce_k = -(target_probs * value_log_probs).sum(dim=-1)            # (K, mb)
    mk = mask.T                                                      # (K, mb)
    return (ce_k * mk).sum() / mk.sum().clamp_min(1.0)


def test_value_loss_mask_prob_one_equals_mean_ce():
    torch.manual_seed(1)
    K, mb, n = 4, 6, 31
    value_logits = torch.randn(K, mb, n)
    target_probs = torch.softmax(torch.randn(mb, n), dim=-1)
    mask = torch.ones(mb, K)

    masked = _masked_value_loss(value_logits, target_probs, mask)
    ce_k = -(target_probs * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1)
    assert torch.allclose(masked, ce_k.mean(), atol=1e-6)


def test_value_loss_hand_set_mask_matches_manual_average():
    K, mb, n = 3, 4, 31
    value_logits = torch.randn(K, mb, n)
    target_probs = torch.softmax(torch.randn(mb, n), dim=-1)
    mask = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])  # (mb, K)

    masked = _masked_value_loss(value_logits, target_probs, mask)
    ce_k = -(target_probs * torch.log_softmax(value_logits, dim=-1)).sum(dim=-1)  # (K, mb)
    mk = mask.T
    expected = (ce_k * mk).sum() / mk.sum()
    assert torch.allclose(masked, expected, atol=1e-6)


# 6. priors frozen and partitioning -----------------------------------------
def test_priors_frozen_and_excluded_from_critic_params():
    agent, _ = _agent()
    for prior in agent.critic_priors:
        for p in prior.parameters():
            assert p.requires_grad is False

    crit_ids = {id(p) for p in agent.critic_parameters()}
    # No prior param is in the critic param group.
    for prior in agent.critic_priors:
        for p in prior.parameters():
            assert id(p) not in crit_ids
    # Every trainable critic-head param IS in the critic param group.
    for head in agent.critic_heads:
        for p in head.parameters():
            assert id(p) in crit_ids


def test_get_value_returns_ensemble_logits():
    agent, args = _agent()
    x = torch.randn(5, 4)
    logits = agent.get_value(x)
    assert logits.shape == (args.n_critics, 5, args.num_bins)


def test_get_action_and_value_returns_ensemble_logits():
    agent, args = _agent()
    x = torch.randn(5, 4)
    _, z, logp, ent, value_logits = agent.get_action_and_value(x)
    assert value_logits.shape == (args.n_critics, 5, args.num_bins)
    assert logp.shape == (5,)
    assert ent.shape == (5,)


def test_prior_scale_zero_disables_prior():
    # prior_scale==0 => _critic_logits equals the bare per-head-trunk logits.
    agent, args = _agent(prior_scale=0.0)
    x = torch.randn(5, 4)
    with_helper = agent._critic_logits(x)
    bare = torch.stack(
        [head(trunk(x)) for head, trunk in zip(agent.critic_heads, agent.critic_trunks)],
        dim=0,
    )
    assert torch.allclose(with_helper, bare, atol=1e-6)


def test_prior_changes_logits_when_enabled():
    agent, args = _agent(prior_scale=2.0)
    x = torch.randn(5, 4)
    with_helper = agent._critic_logits(x)
    bare = torch.stack(
        [head(trunk(x)) for head, trunk in zip(agent.critic_heads, agent.critic_trunks)],
        dim=0,
    )
    assert not torch.allclose(with_helper, bare, atol=1e-4)


# 6b. v2 independent critic trunks ------------------------------------------
def test_independent_trunks_are_distinct_modules_with_own_params():
    agent, args = _agent(independent_critic_trunks=True)
    # K distinct ThinkTrunk modules.
    assert len(agent.critic_trunks) == args.n_critics
    for trunk in agent.critic_trunks:
        assert isinstance(trunk, ThinkTrunk)
    trunk_ids = [id(t) for t in agent.critic_trunks]
    assert len(set(trunk_ids)) == args.n_critics      # all distinct objects
    # Disjoint parameter tensors across trunks (no sharing).
    seen = set()
    for trunk in agent.critic_trunks:
        for p in trunk.parameters():
            assert id(p) not in seen
            seen.add(id(p))
    # critic_parameters() includes EVERY param of ALL K trunks.
    crit_ids = {id(p) for p in agent.critic_parameters()}
    for trunk in agent.critic_trunks:
        for p in trunk.parameters():
            assert id(p) in crit_ids
    # No shared trunk attribute in the independent case.
    assert not hasattr(agent, "trunk")
    assert not hasattr(agent, "critic_trunk")


def test_independent_trunks_actor_critic_param_groups_disjoint():
    # With independent trunks the actor has its OWN trunk: zero overlap with critics
    # (so dual-backward's trunk re-add logic has nothing to re-add).
    agent, _ = _agent(independent_critic_trunks=True)
    actor_ids = {id(p) for p in agent.actor_parameters()}
    crit_ids = {id(p) for p in agent.critic_parameters()}
    assert actor_ids.isdisjoint(crit_ids)


def test_independent_trunks_fresh_heads_disagree_nonzero_vepi():
    # The whole point of v2: freshly-initialized independent-trunk heads produce
    # DIFFERENT values on random obs => v_epi > 0 (v1's shared trunk could not
    # guarantee this — identical features collapse disagreement).
    torch.manual_seed(7)
    support = torch.linspace(-10.0, 10.0, 31)
    agent, _ = _agent(independent_critic_trunks=True, num_bins=31, prior_scale=0.0)
    x = torch.randn(32, 4)
    with torch.no_grad():
        logits = agent.get_value(x)               # (K, B, n)
        _, v_epi = decode_value(logits, support)
    # Heads with different trunks must disagree somewhere (mean std strictly > 0),
    # even with the prior disabled (so it's trunk+init diversity, not the prior).
    assert v_epi.mean().item() > 1e-4
    # And per-head values are genuinely different (not a numerical fluke).
    probs = torch.softmax(logits, dim=-1)
    v_k = (probs * support).sum(dim=-1)           # (K, B)
    assert not torch.allclose(v_k[0], v_k[1], atol=1e-3)


def test_shared_trunk_fallback_reproduces_v1_structure():
    # independent_critic_trunks=False falls back to v1's shared-trunk path.
    agent, args = _agent(independent_critic_trunks=False)
    assert not hasattr(agent, "critic_trunks")
    assert hasattr(agent, "trunk")                # share_backbone default True
    x = torch.randn(5, 4)
    logits = agent.get_value(x)
    assert logits.shape == (args.n_critics, 5, args.num_bins)
    # All heads share ONE trunk's features => bare logits use the single trunk.
    shared = agent.trunk(x)
    bare = torch.stack([head(shared) for head in agent.critic_heads], dim=0)
    # With prior_scale default >0 they differ only by the prior; subtract it.
    with_helper = agent._critic_logits(x)
    prior_term = torch.stack(
        [agent.prior_scale * prior(x) for prior in agent.critic_priors], dim=0
    )
    assert torch.allclose(with_helper - prior_term, bare, atol=1e-5)


# 7. defaults ---------------------------------------------------------------
def test_defaults():
    a = Args()
    assert a.n_critics == 4
    assert a.epistemic_kappa == 1.0
    assert a.prior_scale == 2.0
    assert a.epistemic_anneal is True
    assert a.bootstrap_mask_prob == 0.8
    assert a.independent_critic_trunks is True


# 8. rank_corr probe -------------------------------------------------------
def test_rank_corr_identical_is_one():
    torch.manual_seed(3)
    a = torch.randn(500)
    assert abs(rank_corr(a, a.clone()) - 1.0) < 1e-6
    assert abs(rank_corr(a, 2.0 * a + 5.0) - 1.0) < 1e-6


def test_rank_corr_negated_is_minus_one():
    torch.manual_seed(4)
    a = torch.randn(500)
    assert abs(rank_corr(a, -a) - (-1.0)) < 1e-6


def test_rank_corr_constant_input_is_finite():
    a = torch.zeros(50)
    c = rank_corr(a, torch.randn(50))
    assert np.isfinite(c)


def test_kappa0_policyadv_rank_corr_is_one_kappa_pos_below_one():
    torch.manual_seed(5)
    T, B = 16, 8
    rewards = torch.randn(T, B)
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)
    values = torch.randn(T, B)
    values_epi = torch.rand(T, B) * 3.0
    boot_epi = torch.rand(B) * 3.0
    next_value = torch.randn(1, B)
    gamma, lam = 0.99, 0.95

    advantages = optimistic_policy_gae(
        rewards, dones, next_done, values, torch.zeros(T, B), torch.zeros(B),
        next_value, kappa=0.0, gamma=gamma, lam=lam,
    )
    policy_adv0 = optimistic_policy_gae(
        rewards, dones, next_done, values, values_epi, boot_epi,
        next_value, kappa=0.0, gamma=gamma, lam=lam,
    )
    policy_advk = optimistic_policy_gae(
        rewards, dones, next_done, values, values_epi, boot_epi,
        next_value, kappa=1.0, gamma=gamma, lam=lam,
    )

    corr0 = rank_corr(policy_adv0.reshape(-1), advantages.reshape(-1))
    corrk = rank_corr(policy_advk.reshape(-1), advantages.reshape(-1))
    assert abs(corr0 - 1.0) < 1e-6
    assert corrk < 1.0
    assert corrk > 0.0


def test_inherited_value_defaults_match_base():
    base = D4Args()
    a = Args()
    assert a.num_bins == base.num_bins == 511
    assert a.v_min == base.v_min == -10.0
    assert a.v_max == base.v_max == 10.0
    assert a.value_sigma_to_bin_ratio == base.value_sigma_to_bin_ratio == 2.0
    assert a.critic_init_tau == base.critic_init_tau == 0.5
    assert a.actor_dist == base.actor_dist == "beta"

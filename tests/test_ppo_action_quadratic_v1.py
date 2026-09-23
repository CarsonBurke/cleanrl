"""CUDA algebra/optimizer regression tests; execute through mlq, never a training smoke run."""
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Beta, kl_divergence

from cleanrl.ppo_continuous_action_normres_action_quadratic_v1 import (
    ActorTrustRegion,
    Agent,
    Args,
    action_features,
    beta_feature_means,
    beta_kl,
    beta_kl_reference,
    gradient_diagnostics,
    ppo_loss,
)


class ActionQuadraticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Run these CUDA checks through mlq on a CUDA host")
        cls.device = torch.device("cuda")
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1)

    def test_compiled_kl_matches_distribution_kl_across_candidates(self):
        old_alpha = torch.tensor([1.1, 2.0, 8.0, 20.0], device=self.device)
        old_beta = torch.tensor([3.0, 1.2, 12.0, 15.0], device=self.device)
        reference = beta_kl_reference(old_alpha, old_beta)
        compiled_kl = torch.compile(beta_kl, fullgraph=True, mode="reduce-overhead")
        for scale in (1.0, 1.1, 0.8):
            torch.compiler.cudagraph_mark_step_begin()
            alpha, beta = old_alpha * scale, old_beta / scale
            actual = compiled_kl(alpha, beta, reference).clone()
            expected = kl_divergence(Beta(old_alpha, old_beta), Beta(alpha, beta))
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=5e-4)

    def quadrature(self):
        nodes, weights = np.polynomial.legendre.leggauss(96)
        nodes = torch.as_tensor((nodes + 1) / 2, device=self.device, dtype=torch.float64)
        weights = torch.as_tensor(weights / 2, device=self.device, dtype=torch.float64)
        actions = torch.cartesian_prod(nodes, nodes)
        weights = (weights[:, None] * weights[None, :]).flatten()
        pair_i = torch.tensor([0], device=self.device)
        pair_j = torch.tensor([1], device=self.device)
        alpha = torch.tensor([2.3, 3.2], device=self.device, dtype=torch.float64)
        beta = torch.tensor([3.4, 2.7], device=self.device, dtype=torch.float64)
        mass = weights * Beta(alpha, beta).log_prob(actions).sum(-1).exp()
        return actions, mass, alpha, beta, pair_i, pair_j

    def test_analytic_centering_and_corrected_gradient(self):
        actions, mass, old_alpha, old_beta, pair_i, pair_j = self.quadrature()
        features = action_features(actions, pair_i, pair_j)
        means = beta_feature_means(old_alpha, old_beta, pair_i, pair_j)
        torch.testing.assert_close((mass[:, None] * features).sum(0), means, atol=2e-8, rtol=2e-7)
        coefficients = torch.tensor([1.7, -0.8, 0.9, -1.2, 2.1], device=self.device, dtype=torch.float64)
        control = ((features - means) * coefficients).sum(-1)
        torch.testing.assert_close((mass * control).sum(), torch.zeros((), device=self.device, dtype=torch.float64), atol=2e-8, rtol=0)
        # Deliberately nonquadratic advantage: the learned control need not be correct.
        advantage = torch.sin(4 * actions[:, 0]) + actions[:, 1].pow(3) - actions.prod(-1)
        advantage = advantage - (mass * advantage).sum()
        alpha = old_alpha.clone().requires_grad_()
        beta = old_beta.clone().requires_grad_()
        logprob = Beta(alpha, beta).log_prob(actions).sum(-1)
        ordinary = (mass * logprob * advantage).sum()
        expected_control = ((beta_feature_means(alpha, beta, pair_i, pair_j) - means) * coefficients).sum()
        corrected = (mass * logprob * (advantage - control)).sum() + expected_control
        ordinary_gradient = torch.autograd.grad(ordinary, (alpha, beta), retain_graph=True)
        corrected_gradient = torch.autograd.grad(corrected, (alpha, beta))
        for expected, actual in zip(ordinary_gradient, corrected_gradient):
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)

    def make_batch(self):
        envs = SimpleNamespace(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            single_action_space=gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
        )
        agent = Agent(envs).to(self.device)
        observations = torch.randn(16, 3, device=self.device)
        with torch.no_grad():
            alpha, beta, values = agent.get_policy_and_value(observations)
            native = Beta(alpha, beta).sample()
            logprobs = agent.action_logprob(alpha, beta, native)
            means = beta_feature_means(alpha, beta, agent.pair_i, agent.pair_j)
        advantages = torch.randn(16, device=self.device)
        coefficients = torch.randn(16, 5, device=self.device)
        return agent, observations, native, logprobs, advantages, values.flatten(), means, coefficients

    def test_ordinary_actor_matches_scalar_ppo_and_frozen_inputs_do_not_leak(self):
        agent, observations, native, logprobs, advantages, values, means, coefficients = self.make_batch()
        means.requires_grad_()
        coefficients.requires_grad_()
        for mode in ("ordinary", "corrected"):
            args = Args(actor_update=mode)
            agent.zero_grad(set_to_none=True)
            loss, _ = ppo_loss(
                agent, observations, native, logprobs, advantages, advantages + values, values,
                means, coefficients, args,
            )
            actor_gradients = torch.autograd.grad(loss, tuple(agent.actor.parameters()), retain_graph=True)
            if mode == "ordinary":
                alpha, beta, _ = agent.get_policy_and_value(observations)
                ratio = (agent.action_logprob(alpha, beta, native) - logprobs).exp()
                expected_loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2)).mean()
                expected_gradients = torch.autograd.grad(expected_loss, tuple(agent.actor.parameters()))
                for actual, expected in zip(actor_gradients, expected_gradients):
                    torch.testing.assert_close(actual, expected)
            loss.backward()
            self.assertIsNone(means.grad)
            self.assertIsNone(coefficients.grad)
            action_gradient = agent.critic.action_head.weight.grad
            assert action_gradient is not None
            self.assertGreater(float(action_gradient.norm()), 0)

    def test_zero_control_recovers_ordinary_loss_and_gradient_diagnostics(self):
        agent, observations, native, logprobs, advantages, values, means, coefficients = self.make_batch()
        coefficients.zero_()
        losses = []
        for mode in ("ordinary", "corrected"):
            loss, _ = ppo_loss(
                agent, observations, native, logprobs, advantages, advantages + values, values,
                means, coefficients, Args(actor_update=mode),
            )
            losses.append(loss)
        torch.testing.assert_close(losses[0], losses[1])
        diagnostics = gradient_diagnostics(agent, observations, native, advantages, means, coefficients)
        torch.testing.assert_close(diagnostics["gradient/ordinary_variance_trace"], diagnostics["gradient/corrected_variance_trace"])
        self.assertEqual(float(diagnostics["gradient/analytic_mean_norm"]), 0)

    def test_production_corrected_gradient_before_and_after_residual_clipping(self):
        agent, observations, native, logprobs, advantages, values, means, coefficients = self.make_batch()
        for shifted in (False, True):
            if shifted:
                with torch.no_grad():
                    agent.actor[-1].bias.add_(torch.tensor([2.0, -2.0, -2.0, 2.0], device=self.device))
            actual_loss, _ = ppo_loss(
                agent, observations, native, logprobs, advantages, advantages + values, values,
                means, coefficients, Args(actor_update="corrected"),
            )
            actual = torch.autograd.grad(actual_loss, tuple(agent.actor.parameters()))
            alpha, beta, _ = agent.get_policy_and_value(observations)
            distribution = Beta(alpha, beta)
            current_logprobs = agent.action_logprob(alpha, beta, native)
            ratio = (current_logprobs - logprobs).exp()
            y = 2 * native - 1
            features = torch.stack((y[:, 0], y[:, 1], y[:, 0] ** 2, y[:, 1] ** 2, y[:, 0] * y[:, 1]), -1)
            residual = advantages - (coefficients * (features - means)).sum(-1)
            active = ((residual >= 0) & (ratio < 1.2)) | ((residual < 0) & (ratio > 0.8))
            if shifted:
                self.assertTrue(bool(active.any()))
                self.assertTrue(bool((~active).any()))
            mean_y = 2 * distribution.mean - 1
            second_y = 4 * distribution.variance + mean_y.square()
            expected_features = torch.stack(
                (mean_y[:, 0], mean_y[:, 1], second_y[:, 0], second_y[:, 1], mean_y[:, 0] * mean_y[:, 1]), -1
            )
            # Independent derivative: clipped score term plus exact integrated action effect.
            expected_loss = -(current_logprobs * (ratio * residual * active).detach()).mean()
            expected_loss -= (coefficients * (expected_features - means)).sum(-1).mean()
            expected = torch.autograd.grad(expected_loss, tuple(agent.actor.parameters()))
            for actual_gradient, expected_gradient in zip(actual, expected):
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-7, rtol=2e-5)

    def test_accepted_actor_respects_exact_beta_kl(self):
        actor = torch.nn.Linear(1, 2, device=self.device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=4.0, fused=True)
        observations = torch.ones(8, 1, device=self.device)
        def distribution():
            alpha, beta = (torch.nn.functional.softplus(actor(observations)) + 1).chunk(2, -1)
            return Beta(alpha, beta)
        with torch.no_grad():
            old = distribution()
        trust = ActorTrustRegion(actor, optimizer, budget=0.002, max_backtracks=12)
        (-distribution().mean.mean()).backward()
        trust.snapshot()
        optimizer.step()
        kl, fraction, _ = trust.accept(lambda: kl_divergence(old, distribution()).mean())
        self.assertGreater(fraction, 0)
        self.assertLess(fraction, 1)
        self.assertLessEqual(float(kl), 0.002)

    def test_rejected_actor_restores_adam_without_undoing_critic(self):
        actor = torch.nn.Linear(1, 1, device=self.device)
        critic = torch.nn.Linear(1, 1, device=self.device)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.1, fused=True)
        x = torch.ones(2, 1, device=self.device)
        # Cover both uninitialized and populated Adam state.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (actor(x).square().mean() + critic(x).square().mean()).backward()
            actor_before = [p.detach().clone() for p in actor.parameters()]
            critic_before = [p.detach().clone() for p in critic.parameters()]
            state_before = [{k: v.clone() for k, v in optimizer.state.get(p, {}).items()} for p in actor.parameters()]
            trust = ActorTrustRegion(actor, optimizer, budget=0.001, max_backtracks=0)
            trust.snapshot()
            optimizer.step()
            _, fraction, _ = trust.accept(lambda: torch.ones((), device=self.device))
            self.assertEqual(fraction, 0)
            for parameter, expected, state in zip(actor.parameters(), actor_before, state_before):
                torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
                self.assertEqual(set(optimizer.state.get(parameter, {})), set(state))
                for key, value in state.items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value, atol=0, rtol=0)
            self.assertTrue(any(not torch.equal(p, previous) for p, previous in zip(critic.parameters(), critic_before)))
            optimizer.step()  # Populate actor state for the next rejection case.


if __name__ == "__main__":
    unittest.main()

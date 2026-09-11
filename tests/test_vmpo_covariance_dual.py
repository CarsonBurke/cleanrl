"""Covariance-dual regressions; CUDA execution must go through mlq."""
import pytest
import torch

from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.runtime import configure_runtime
from cleanrl.vmpo.ppo_continuous_action_vmpo_paper_v60_solved_estep import vmpo_loss as reference_loss
from cleanrl.vmpo.ppo_continuous_action_vmpo_paper_v62_covariance_dual import (
    Agent, Args, DUAL_FLOOR, update_covariance_dual, validate_args, vmpo_loss,
)

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def cuda_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


def test_compiled_budget_relative_response_and_floor_recovery():
    epsilon = Args().epsilon_alpha_covariance
    alpha = torch.tensor(1.0, device="cuda")
    kl = torch.tensor(2 * epsilon, device="cuda", requires_grad=True)
    update = graph_compile(update_covariance_dual)
    for _ in range(100):
        update(alpha, kl, epsilon, 100)
    torch.testing.assert_close(alpha, torch.tensor(2.0, device="cuda"), rtol=0, atol=5e-6)
    assert not alpha.requires_grad and kl.grad is None
    with torch.no_grad():
        kl.fill_(epsilon)
    before = alpha.clone()
    update(alpha, kl, epsilon, 100)
    torch.testing.assert_close(alpha, before, rtol=0, atol=0)
    with torch.no_grad():
        kl.zero_()
    for _ in range(210):
        update(alpha, kl, epsilon, 100)
    torch.testing.assert_close(alpha, torch.tensor(DUAL_FLOOR, device="cuda"), rtol=0, atol=0)
    with torch.no_grad():
        kl.fill_(20 * epsilon)
    update(alpha, kl, epsilon, 100)
    torch.testing.assert_close(alpha, torch.tensor(0.19 + DUAL_FLOOR, device="cuda"), rtol=1e-6, atol=1e-7)


def test_compiled_actor_and_mean_dual_gradients_match_reference_after_covariance_mutation():
    args = validate_args(Args(num_envs=8, num_steps=3, num_replicas=2))
    agent = Agent(17, 6).cuda()
    alpha_mean = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    alpha_covariance = torch.tensor(1.0, device="cuda")
    obs = torch.randn(args.batch_size, 17, device="cuda")
    with torch.no_grad():
        mean, std = agent.policy(obs)
        old_mean, old_std = mean + 0.04, std * 1.04
        actions = old_mean + old_std * torch.randn_like(old_mean)
    advantages = torch.linspace(-2, 2, args.batch_size, device="cuda")
    targets = torch.linspace(-1, 1, args.batch_size, device="cuda")

    def loss_fn(adv):
        return vmpo_loss(agent, alpha_mean, alpha_covariance, obs, actions,
                         old_mean, old_std, adv, targets, args)

    compiled = torch.compile(loss_fn, fullgraph=True, mode="reduce-overhead")
    parameters = list(agent.parameters())
    observed = []
    for coefficient in (1.0, 7.0):
        alpha_covariance.fill_(coefficient)
        reference_duals = torch.nn.Parameter(torch.tensor([1.0, coefficient], device="cuda"))
        expected_loss, expected_metrics = reference_loss(
            agent, reference_duals, obs, actions, old_mean, old_std, advantages, targets, args)
        expected_grads = torch.autograd.grad(expected_loss, [*parameters, reference_duals])
        torch.compiler.cudagraph_mark_step_begin()
        actual_loss, actual_metrics = compiled(advantages)
        actual_grads = torch.autograd.grad(actual_loss, [*parameters, alpha_mean])
        torch.testing.assert_close(actual_metrics[4], expected_metrics[4], rtol=3e-4, atol=2e-6)
        for actual, expected in zip(actual_grads[:-1], expected_grads[:-1]):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5)
        torch.testing.assert_close(actual_grads[-1], expected_grads[-1][0], rtol=3e-4, atol=2e-5)
        observed.append(torch.cat([gradient.flatten() for gradient in actual_grads[:-1]]).clone())
    # Replaying the compiled loss must consume the new multiplier, not capture its initial value.
    assert (observed[1] - observed[0]).norm() > 0.1

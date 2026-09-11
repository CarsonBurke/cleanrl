"""Shared replica-temperature regression tests; CUDA execution requires mlq."""
import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from cleanrl.vmpo.ppo_continuous_action_vmpo_paper_v60_solved_estep import (
    Agent, Args, DUAL_FLOOR, replica_estep, validate_args, vmpo_loss,
)

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def cuda_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


def test_shared_temperature_constrains_average_not_each_replica():
    base = torch.linspace(-2, 2, 312, device="cuda")
    advantages = torch.stack((base, 100 * base))
    weights, _, kl, _, selected, eta = replica_estep(advantages, 0.01)
    torch.testing.assert_close(kl.mean(), torch.tensor(0.01, device="cuda"), atol=2e-6, rtol=0)
    # Independently solving each replica would wrongly spend .01 in BOTH.
    assert kl[0] < 1e-4
    assert kl[1] > 0.019
    assert eta > 1
    assert (selected == 156).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, device="cuda"), atol=1e-6, rtol=0)
    assert (weights[:, :156] == 0).all()


def test_solver_preserves_weights_across_reward_units_and_common_shifts():
    advantages = torch.randn(8, 312, device="cuda", dtype=torch.float64)
    reference = replica_estep(advantages, 0.01)
    for scale, shift in ((1e-4, 0.0), (1e4, 0.0), (1.0, 1000.0)):
        actual = replica_estep(advantages * scale + shift, 0.01)
        torch.testing.assert_close(actual[0], reference[0], atol=1e-10, rtol=2e-7)
        torch.testing.assert_close(actual[2], reference[2], atol=1e-9, rtol=1e-6)
        torch.testing.assert_close(actual[-1], reference[-1] * scale, atol=1e-9, rtol=2e-7)


def test_flat_elites_have_slack_at_floor_and_no_target_gradients():
    advantages = torch.ones(8, 312, device="cuda", requires_grad=True)
    weights, loss, kl, ess, count, eta = replica_estep(advantages, 0.01)
    assert not weights.requires_grad and not loss.requires_grad and not eta.requires_grad
    torch.testing.assert_close(eta, torch.tensor(DUAL_FLOOR, device="cuda"), rtol=1e-6, atol=0)
    torch.testing.assert_close(kl, torch.zeros(8, device="cuda"), atol=1e-6, rtol=0)
    torch.testing.assert_close(ess, count.float(), rtol=1e-6, atol=0)
    torch.testing.assert_close((weights / 8).square().sum().reciprocal(),
                               torch.tensor(2496., device="cuda"), rtol=1e-6, atol=0)


def test_compiled_joint_update_is_reward_unit_invariant_and_duals_remain_live():
    args = validate_args(Args(num_envs=8, num_steps=3, num_replicas=2))
    agent = Agent(17, 6).cuda()
    duals = torch.nn.Parameter(torch.ones(2, device="cuda"))
    obs = torch.randn(args.batch_size, 17, device="cuda")
    with torch.no_grad():
        mean, std = agent.policy(obs)
        old_mean, old_std = mean + 0.4, std * 1.2
        actions = old_mean + old_std * torch.randn_like(old_mean)
    advantage = torch.linspace(-2, 2, args.batch_size, device="cuda")
    targets = torch.linspace(-1, 1, args.batch_size, device="cuda")

    def loss_fn(adv):
        return vmpo_loss(agent, duals, obs, actions, old_mean, old_std, adv, targets, args)

    compiled = torch.compile(loss_fn, fullgraph=True, mode="reduce-overhead")
    parameters = [*agent.parameters(), duals]
    reference_loss, reference_metrics = loss_fn(advantage)
    reference_grads = torch.autograd.grad(reference_loss, parameters)
    assert (reference_grads[-1] < 0).all()
    for scale in (1.0, 1000.0):
        torch.compiler.cudagraph_mark_step_begin()
        actual_loss, actual_metrics = compiled(advantage * scale)
        actual_grads = torch.autograd.grad(actual_loss, parameters)
        torch.testing.assert_close(actual_loss, reference_loss, rtol=2e-5, atol=1e-5)
        torch.testing.assert_close(actual_metrics[-1], reference_metrics[-1] * scale, rtol=3e-4, atol=1e-4)
        torch.testing.assert_close(actual_metrics[6], torch.tensor(0.01, device="cuda"), rtol=0, atol=2e-6)
        for actual, expected in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5)

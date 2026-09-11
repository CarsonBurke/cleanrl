import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.vmpo.ppo_continuous_action_vmpo_paper_v61_rollout_popart import Agent

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("rate", [1e-4, 1.0])
def test_compiled_rollout_moments_preserve_raw_predictions(rate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    agent = Agent(3, 2).cuda()
    with torch.no_grad():
        agent.value_head.weight.normal_(std=0.1)
        agent.value_head.bias.fill_(0.7)
    observations = torch.randn(9, 3, device="cuda")
    # Within-rollout variation must not be mistaken for between-rollout variance.
    returns = torch.tensor([[-8., 3., -7., 4.], [12., 5., 3., 8.]], device="cuda")
    mean, second = 0.0, 1.0
    update = graph_compile(agent.update_popart)
    optimizer = torch.optim.SGD(agent.value_head.parameters(), lr=0.01)
    for offset in (0.0, 7.0):
        before = agent.value(observations).detach().clone()
        targets = returns + offset
        for sample in [2.0 + offset, 4.0 + offset, -2.0 + offset, 6.0 + offset]:
            mean = (1 - rate) * mean + rate * sample
            second = (1 - rate) * second + rate * sample * sample
        normalized = update(targets, rate, 0.01, 1e6)
        torch.testing.assert_close(agent.popart_mean, torch.tensor(mean, device="cuda"), rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(agent.popart_sq_mean, torch.tensor(second, device="cuda"), rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(agent.value(observations), before, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(normalized * agent.popart_std + agent.popart_mean, targets, rtol=2e-6, atol=2e-6)
        # The next statistics transition must preserve the fitted, not initial, critic.
        optimizer.zero_grad(set_to_none=True)
        (agent.value(observations) - 2.0).square().mean().backward()
        optimizer.step()

"""Raw-unit variance matching must rescale CE, not optimize its own gain."""
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_variance_matched_v17 import categorical_value_terms
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.two_hot import DreamerTwoHotSupport


def test_compiled_raw_batch_gain_scales_gradients_and_squared_units():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = DreamerTwoHotSupport(101, 20000, device="cuda")
    torch.manual_seed(1)
    logits = torch.randn(2, 101, device="cuda", requires_grad=True)
    labels = torch.randn(2, 101, device="cuda").softmax(-1).requires_grad_()
    with torch.no_grad():
        probabilities = logits.softmax(-1)
        means = (probabilities.double() * histogram.support.double()).sum(-1, keepdim=True)
        gain = (probabilities.double() * (histogram.support.double() - means).square()).sum(-1).mean().float()
        expected_gradient = gain * (probabilities - labels) / labels.shape[0]
    loss_fn = torch.compile(categorical_value_terms, fullgraph=True,
                            options={"triton.cudagraphs": False})
    loss, _, actual_gain = loss_fn(logits, labels, histogram)
    loss.backward()
    torch.testing.assert_close(actual_gain, gain)
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=1e-4, atol=0.1)
    assert labels.grad is None
    previous_loss = loss.detach().clone()
    histogram.support.mul_(2.0)
    logits.grad = None
    loss, _, _ = loss_fn(logits, labels, histogram)
    loss.backward()
    torch.testing.assert_close(loss, 4 * previous_loss)
    torch.testing.assert_close(logits.grad, 4 * expected_gradient, rtol=1e-4, atol=0.4)


def test_matching_distribution_has_no_variance_collapse_gradient():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = DreamerTwoHotSupport(5, 20000, device="cuda").double()
    labels = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1], [0.05, 0.1, 0.15, 0.2, 0.5]],
                          device="cuda", dtype=torch.float64)
    logits = labels.log().requires_grad_()
    loss, _, _ = categorical_value_terms(logits, labels, histogram)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)

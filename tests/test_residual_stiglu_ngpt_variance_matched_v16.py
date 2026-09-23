"""Variance matching must scale CE gradients, not reward variance collapse."""
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_variance_matched_v16 import categorical_value_terms
from cleanrl.shared.hl_gauss_std import StandardizedHistogram
from cleanrl.shared.runtime import configure_runtime


def test_compiled_batch_gain_scales_each_sample_equally_and_tracks_squared_units():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = StandardizedHistogram(5, half_span=5.0, device="cuda")
    histogram.scale.fill_(2.0)
    logits = torch.tensor([[-3., -1., 0., 1., 2.], [-4., -2., 3., -2., -4.]],
                          device="cuda", requires_grad=True)
    labels = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1]],
                          device="cuda", requires_grad=True)
    with torch.no_grad():
        probabilities = logits.softmax(-1)
        means = (probabilities * histogram.centers).sum(-1, keepdim=True)
        variances = (probabilities * (histogram.centers - means).square()).sum(-1)
        gain = histogram.scale.square() * variances.mean()
        expected_gradient = gain * (probabilities - labels) / labels.shape[0]
    loss_fn = torch.compile(categorical_value_terms, fullgraph=True,
                            options={"triton.cudagraphs": False})
    loss, _, _ = loss_fn(logits, labels, histogram)
    loss.backward()
    torch.testing.assert_close(logits.grad, expected_gradient)
    assert labels.grad is None
    previous_loss = loss.detach().clone()
    histogram.scale.mul_(2.0)
    logits.grad = None
    loss, _, _ = loss_fn(logits, labels, histogram)
    loss.backward()
    torch.testing.assert_close(loss, 4 * previous_loss)
    torch.testing.assert_close(logits.grad, 4 * expected_gradient)


def test_matching_distribution_is_stationary_despite_positive_entropy_floor():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = StandardizedHistogram(5, half_span=5.0, device="cuda")
    histogram.scale.fill_(3.0)
    labels = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1], [0.05, 0.1, 0.15, 0.2, 0.5]], device="cuda")
    logits = labels.log().requires_grad_()
    loss, _, _ = categorical_value_terms(logits, labels, histogram)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=2e-6, rtol=0)

"""Guard readout-coordinate equivalence and the requested optimization boundaries."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_readout_geometry_v20 import (
    Agent, Args, apply_gradient_clipping, categorical_value_terms,
)
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.two_hot import DreamerTwoHotSupport


def test_plain_ce_removes_only_detached_variance_scaling():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = DreamerTwoHotSupport(5, 2, device="cuda", spacing="linear")
    labels = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]], device="cuda", requires_grad=True)
    logits = torch.tensor([[0.4, -0.2, 0.1, 0.5, -0.3]], device="cuda", requires_grad=True)
    loss_fn = torch.compile(categorical_value_terms, fullgraph=True,
                            options={"triton.cudagraphs": False})
    plain, _, _, _ = loss_fn(logits, labels, histogram, False)
    plain.backward()
    plain_gradient = logits.grad.clone()
    torch.testing.assert_close(plain, -(labels.detach() * logits.log_softmax(-1)).sum())
    torch.testing.assert_close(plain_gradient, logits.detach().softmax(-1) - labels.detach())
    logits.grad = None
    scaled, _, gain, _ = loss_fn(logits, labels, histogram, True)
    scaled.backward()
    torch.testing.assert_close(scaled, plain.detach() * gain)
    torch.testing.assert_close(logits.grad, plain_gradient * gain)
    assert labels.grad is None
    assert not gain.requires_grad


def test_relative_coordinates_preserve_function_derivatives_and_checkpoint_reload():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32),
    )
    torch.manual_seed(1)
    direct = Agent(spaces, Args(critic_gain_coordinates="direct")).cuda()
    direct.normalize_matrices()
    torch.manual_seed(1)
    relative = Agent(spaces, Args(critic_gain_coordinates="relative")).cuda()
    relative.normalize_matrices()
    x = torch.randn(32, 17, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    direct_fn = torch.compile(direct.critic, fullgraph=True, options={"triton.cudagraphs": False})
    relative_fn = torch.compile(relative.critic, fullgraph=True, options={"triton.cudagraphs": False})
    direct_logits, relative_logits = direct_fn(x), relative_fn(y)
    torch.testing.assert_close(direct_logits, relative_logits)
    weights = torch.randn_like(direct_logits)
    (direct_logits * weights).sum().backward()
    (relative_logits * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(relative.critic.readout_gain.grad, 0.01 * direct.critic.readout_gain.grad)
    # Loading into the default constructor must retain the effective scale, not
    # silently interpret dimensionless u as the physical gamma.
    restored = Agent(spaces).cuda()
    restored.load_state_dict(relative.state_dict())
    restored_fn = torch.compile(restored.critic, fullgraph=True, options={"triton.cudagraphs": False})
    torch.testing.assert_close(restored_fn(x.detach()), relative_logits.detach())


def test_readout_only_clipping_leaves_other_parameter_updates_untouched():
    actor = torch.nn.Parameter(torch.tensor([2.0, 3.0], device="cuda"))
    trunk = torch.nn.Parameter(torch.tensor([5.0, 7.0], device="cuda"))
    readout = torch.nn.Parameter(torch.tensor([0.01, 0.02], device="cuda"))
    parameters = (actor, trunk, readout)
    initial = [p.detach().clone() for p in parameters]
    (actor.square().sum() + trunk.square().sum() + (1000 * readout).square().sum()).backward()
    actor_gradient, trunk_gradient = actor.grad.clone(), trunk.grad.clone()
    apply_gradient_clipping(parameters, (readout,), 0.5, "readout")
    torch.optim.SGD(parameters, lr=0.01).step()
    torch.testing.assert_close(actor, initial[0] - 0.01 * actor_gradient)
    torch.testing.assert_close(trunk, initial[1] - 0.01 * trunk_gradient)
    torch.testing.assert_close((readout - initial[2]).norm(), torch.tensor(0.005, device="cuda"))

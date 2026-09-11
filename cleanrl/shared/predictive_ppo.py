"""Covariance-free output-score transport for the unchanged scalar/Beta PPO loss.

For current minibatch task T, s(theta) = d L_T / d (logits, values) and
C = J(theta)^T [s(theta) - s(theta_previous)], with both scores detached.
The critic-only mode zeros the actor score difference. For unclipped scalar
MSE this is exact predictive transport; the clipped PPO/Beta head instead uses
its finite output-score difference, not a squared-loss or Fisher tangent.

m_t = beta1 m_(t-1) + (1-beta1) g_t + beta1 (1-beta1**(t-1)) C_t;
v_t is ordinary Adam's EMA of raw g_t**2. No covariance/Jacobian is stored.
Both evaluations use the SAME current minibatch, targets and old policy data.
This corrects model drift only: momentum is not reset and target/distribution
drift across rollouts is not transported. Head scores include entropy and the
signed PPO clipping branches, with PyTorch's original max/clamp tie behavior.
"""

import math

import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch.func import functional_call, grad, vjp


def ppo_head_loss(
    logits,
    values,
    native_actions,
    old_logprobs,
    advantages,
    targets,
    old_values,
    log_action_scale,
    *,
    clip_coef,
    clip_vloss,
    ent_coef,
    vf_coef,
    norm_adv,
):
    """Return v22's scalar loss and detached six-element logging tensor."""
    alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - log_action_scale).sum(-1)
    entropy = (distribution.entropy() + log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    if norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    values = values.view(-1)
    if clip_vloss:
        v_loss_unclipped = (values - targets) ** 2
        v_clipped = old_values + torch.clamp(values - old_values, -clip_coef, clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - targets) ** 2).mean()
    else:
        v_loss = 0.5 * ((values - targets) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
    metrics = torch.stack(
        (pg_loss.detach(), v_loss.detach(), entropy_loss.detach(), old_approx_kl, approx_kl, clipfrac)
    )
    return loss, metrics


def make_gradient_function(agent, args):
    """Build a fullgraph-compilable (params, previous, minibatch...) transform.

    Agent.forward returns raw actor logits and flattened scalar critic values.
    Parameters may be detached views of the live agent; previous must have
    independent storage. Returned named gradients/corrections are detached and
    can be clipped by the caller before the optimizer consumes them.
    """
    mode = args.transport_mode
    if mode not in {"adam", "critic", "both"}:
        raise ValueError(f"unknown transport mode: {mode}")
    clip_coef, clip_vloss = args.clip_coef, args.clip_vloss
    ent_coef, vf_coef, norm_adv = args.ent_coef, args.vf_coef, args.norm_adv

    def head_loss(logits, values, native, old_logprobs, advantages, targets, old_values):
        return ppo_head_loss(
            logits, values, native, old_logprobs, advantages, targets, old_values,
            agent.log_action_scale, clip_coef=clip_coef, clip_vloss=clip_vloss,
            ent_coef=ent_coef, vf_coef=vf_coef, norm_adv=norm_adv,
        )

    head_scores = grad(head_loss, argnums=(0, 1), has_aux=True)

    def gradients(params, previous, observations, native, old_logprobs, advantages, targets, old_values):
        def forward(weights):
            return functional_call(agent, weights, (observations,))

        outputs, pullback = vjp(forward, params)
        task = (native, old_logprobs, advantages, targets, old_values)
        scores, metrics = head_scores(outputs[0].detach(), outputs[1].detach(), *task)
        scores = tuple(score.detach() for score in scores)
        grads = {name: value.detach() for name, value in pullback(scores)[0].items()}
        if mode == "adam":
            # No previous forward, previous head scores or second pullback.
            return grads, {name: torch.zeros_like(value) for name, value in grads.items()}, metrics

        with torch.no_grad():
            previous_outputs = forward(previous)
        previous_scores, _ = head_scores(
            previous_outputs[0].detach(), previous_outputs[1].detach(), *task
        )
        actor_difference = (
            scores[0] - previous_scores[0].detach()
            if mode == "both" else torch.zeros_like(scores[0])
        )
        score_difference = (actor_difference.detach(), (scores[1] - previous_scores[1]).detach())
        corrections = {
            name: value.detach() for name, value in pullback(score_difference)[0].items()
        }
        return grads, corrections, metrics

    return gradients


def _compute_update(params, grads, corrections, moments, variances, step, lr, beta1, beta2, eps):
    """Pure tensor transition: parameter and optimizer input storage stays read-only."""
    next_step = step + 1
    exponent = next_step.to(dtype=params[0].dtype)
    if beta1 == 0:
        old_mass = torch.zeros_like(exponent)
        mass = torch.ones_like(exponent)
    else:
        old_mass = -torch.expm1(math.log(beta1) * (exponent - 1))
        mass = -torch.expm1(math.log(beta1) * exponent)
    variance_mass = (
        torch.ones_like(exponent) if beta2 == 0
        else -torch.expm1(math.log(beta2) * exponent)
    )
    next_params, next_moments, next_variances = [], [], []
    for parameter, gradient, correction, moment, variance in zip(
        params, grads, corrections, moments, variances
    ):
        next_m = beta1 * moment + (1 - beta1) * gradient + beta1 * old_mass * correction
        next_v = beta2 * variance + (1 - beta2) * gradient.square()
        direction = (next_m / mass) / ((next_v / variance_mass).sqrt() + eps)
        next_params.append(parameter - lr * direction)
        next_moments.append(next_m)
        next_variances.append(next_v)
    return next_params, next_moments, next_variances, next_step


class TransportAdam(torch.optim.Optimizer):
    """One fixed named parameter set, owned previous weights and ordinary Adam v.

    Every step requires full gradient and correction dictionaries. A zero
    correction is the Adam control, using exactly the same moment/update path.
    State uses native Optimizer checkpoints; the group's scalar step advances
    all parameters together. Device scalar LR updates do not specialize graphs.

    Only pure compute is compiled, with CUDA graphs disabled. Commit runs
    outside that graph: copy current -> previous BEFORE current -> next, so
    Inductor cannot erase the distinct previous/current weight lifetimes.
    The trainer must not compile step itself around this mutation boundary.
    """

    def __init__(self, named_params_dict, lr, beta1=0.9, beta2=0.999, eps=1e-5, compile=True):
        named_params = dict(named_params_dict)
        if not named_params:
            raise ValueError("TransportAdam requires a nonempty named parameter dictionary")
        params = list(named_params.values())
        if len({id(parameter) for parameter in params}) != len(params):
            raise ValueError("TransportAdam does not allow duplicate parameters")
        for parameter in params:
            if (
                not isinstance(parameter, torch.Tensor)
                or parameter.layout != torch.strided
                or parameter.numel() == 0
                or parameter.dtype not in (torch.float32, torch.float64)
            ):
                raise ValueError("TransportAdam requires nonempty dense FP32 or FP64 parameters")
            if parameter.device != params[0].device or parameter.dtype != params[0].dtype:
                raise ValueError("TransportAdam parameters must share device and dtype")
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("TransportAdam learning rate must be finite and nonnegative")
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("TransportAdam betas must lie in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("TransportAdam epsilon must be finite and positive")
        first = params[0]
        super().__init__(
            [{"params": params, "param_names": list(named_params)}],
            dict(
                lr=torch.tensor(lr, device=first.device, dtype=first.dtype),
                betas=(beta1, beta2), eps=eps,
            ),
        )
        self.param_groups[0]["step"] = torch.zeros((), device=first.device, dtype=torch.int64)
        self.previous = {}
        for name, parameter in named_params.items():
            state = self.state[parameter]
            state["previous"] = parameter.detach().clone()
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
            self.previous[name] = state["previous"]
        self._compute_update = (
            torch.compile(_compute_update, fullgraph=True, options={"triton.cudagraphs": False})
            if compile else _compute_update
        )

    @torch.no_grad()
    def set_lr(self, value):
        """Update the device scalar in place, including exact zero at schedule end."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("TransportAdam learning rate must be finite and nonnegative")
        self.param_groups[0]["lr"].fill_(value)

    def load_state_dict(self, state_dict):
        groups = state_dict["param_groups"]
        if len(groups) != 1 or groups[0].get("param_names") != self.param_groups[0]["param_names"]:
            raise ValueError("TransportAdam checkpoint parameter names/order do not match")
        super().load_state_dict(state_dict)
        group = self.param_groups[0]
        first = group["params"][0]
        group["lr"] = group["lr"].to(device=first.device, dtype=first.dtype)
        group["step"] = group["step"].to(device=first.device, dtype=torch.int64)
        self.previous.clear()
        for name, parameter in zip(group["param_names"], group["params"]):
            state = self.state[parameter]
            if not all(key in state for key in ("previous", "exp_avg", "exp_avg_sq")):
                raise ValueError("TransportAdam checkpoint is missing moment/previous state")
            self.previous[name] = state["previous"]

    @torch.no_grad()
    def step(self, grads, corrections):
        group = self.param_groups[0]
        names, params = group["param_names"], group["params"]
        moments = [self.state[parameter]["exp_avg"] for parameter in params]
        variances = [self.state[parameter]["exp_avg_sq"] for parameter in params]
        beta1, beta2 = group["betas"]
        next_params, next_moments, next_variances, next_step = self._compute_update(
            params, [grads[name] for name in names], [corrections[name] for name in names],
            moments, variances, group["step"], group["lr"], beta1, beta2, group["eps"],
        )
        # Never move these commits into the compiled read-only compute function.
        for name, parameter in zip(names, params):
            self.previous[name].copy_(parameter)
        for parameter, following, moment, next_m, variance, next_v in zip(
            params, next_params, moments, next_moments, variances, next_variances
        ):
            parameter.copy_(following)
            moment.copy_(next_m)
            variance.copy_(next_v)
        group["step"].copy_(next_step)

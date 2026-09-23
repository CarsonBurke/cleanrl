"""Exact delayed-credit derivations, not an optimizer or training experiment.

For one originating log gain a, delta_0 = -eta_0 grad L_0(theta_0) is
fixed and theta_1(a) = theta_0 + exp(a) delta_0. Consequently J_0 = 0,
J_1 = exp(a) delta_0, and every subsequent plain-SGD update transports
J_{t+1} = J_t - eta_t H_t J_t. All Hessians are evaluated at the actual
pre-update theta_t, not at the originating parameters.

Historical utility here is L_eval(theta_T_without_origin) - L_eval(theta_T(a)).
The reference replays the same later examples but omits the originating update;
it is independent of a. Its derivative is -grad L_eval(theta_T(a)) dot J_T.
This is NOT the local finite-utility derivative with a retained displacement:
-grad L_eval(theta + delta) dot delta. Future updates change the displacement's
sensitivity. Neither derivative is the value of finite utility itself.

Only run_audits() allocates CUDA tensors. The caller must queue its execution.
"""

from collections.abc import Callable
import math

import torch
from torch.func import grad, jacrev, jvp


Loss = Callable[[torch.Tensor], torch.Tensor]


def _assert_close(actual, expected, label, *, rtol=2e-10, atol=2e-11):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=label)


def _audit_case(name, initial, losses, learning_rates, evaluation_loss, step_names):
    """Compare matrix-free transport, reverse AD, and independent full replay."""
    assert len(losses) == len(learning_rates) == len(step_names)
    origin_delta = (-learning_rates[0] * grad(losses[0])(initial)).detach()
    log_gain = initial.new_zeros(())

    def trajectory(a):
        theta = initial + a.exp() * origin_delta
        for loss, rate in zip(losses[1:], learning_rates[1:]):
            theta = theta - rate * grad(loss)(theta)
        return theta

    reference = initial.clone()
    for loss, rate in zip(losses[1:], learning_rates[1:]):
        reference = reference - rate * grad(loss)(reference)
    reference_loss = evaluation_loss(reference).detach()

    def utility(a):
        return reference_loss - evaluation_loss(trajectory(a))

    theta = initial.clone()
    sensitivity = torch.zeros_like(initial)
    records = []
    for index, (loss, rate, label) in enumerate(zip(losses, learning_rates, step_names)):
        before = theta
        sensitivity_before = sensitivity
        if index == 0:
            gradient = grad(loss)(before)
            hessian_vector = torch.zeros_like(before)
            injection = log_gain.exp() * origin_delta
            theta = before + injection
            sensitivity = injection
        else:
            # jvp returns BOTH grad L(theta) and H(theta) J; no dense Hessian.
            gradient, hessian_vector = jvp(grad(loss), (before,), (sensitivity_before,))
            injection = torch.zeros_like(before)
            theta = before - rate * gradient
            sensitivity = sensitivity_before - rate * hessian_vector
        theta = theta.detach()
        sensitivity = sensitivity.detach()
        records.append({
            "step": index,
            "example": label,
            "learning_rate": rate,
            "loss_before": loss(before).item(),
            "theta_before": before.tolist(),
            "theta_after": theta.tolist(),
            "gradient": gradient.tolist(),
            "sensitivity_before": sensitivity_before.tolist(),
            "hessian_vector": hessian_vector.tolist(),
            "injected_sensitivity": injection.tolist(),
            "sensitivity_after": sensitivity.tolist(),
        })

    reverse_sensitivity = jacrev(trajectory)(log_gain)
    evaluation_gradient = grad(evaluation_loss)(theta)
    forward_credit = -torch.dot(evaluation_gradient, sensitivity)
    reverse_credit = grad(utility)(log_gain)
    naive_credit = -torch.dot(evaluation_gradient, origin_delta)
    immediate_credit = -torch.dot(grad(evaluation_loss)(initial + origin_delta), origin_delta)
    _assert_close(theta, trajectory(log_gain), f"{name}: forward trajectory")
    _assert_close(sensitivity, reverse_sensitivity, f"{name}: complete reverse sensitivity")
    _assert_close(forward_credit, reverse_credit, f"{name}: complete reverse credit")
    finite_differences = []
    for epsilon in (1e-4, 1e-5, 1e-6):
        plus = trajectory(log_gain + epsilon)
        minus = trajectory(log_gain - epsilon)
        finite_sensitivity = (plus - minus) / (2 * epsilon)
        # Subtract losses directly to avoid cancelling the fixed reference twice.
        finite_credit = (evaluation_loss(minus) - evaluation_loss(plus)) / (2 * epsilon)
        _assert_close(finite_sensitivity, sensitivity, f"{name}: finite-difference sensitivity {epsilon}",
                      rtol=2e-6, atol=2e-8)
        _assert_close(finite_credit, forward_credit, f"{name}: finite-difference credit {epsilon}",
                      rtol=2e-6, atol=2e-8)
        finite_differences.append({
            "epsilon": epsilon,
            "sensitivity": finite_sensitivity.tolist(),
            "credit": finite_credit.item(),
            "sensitivity_max_absolute_error": (finite_sensitivity - sensitivity).abs().max().item(),
            "credit_absolute_error": (finite_credit - forward_credit).abs().item(),
        })

    return {
        "name": name,
        "origin_log_gain": log_gain.item(),
        "origin_delta_held_fixed": origin_delta.tolist(),
        "initial_theta": initial.tolist(),
        "final_theta": theta.tolist(),
        "without_origin_final_theta": reference.tolist(),
        "final_evaluation_loss": evaluation_loss(theta).item(),
        "without_origin_evaluation_loss": reference_loss.item(),
        "finite_historical_utility": utility(log_gain).item(),
        "immediate_finite_utility": (evaluation_loss(initial) - evaluation_loss(initial + origin_delta)).item(),
        "immediate_utility_derivative": immediate_credit.item(),
        "evaluation_gradient": evaluation_gradient.tolist(),
        "forward_sensitivity": sensitivity.tolist(),
        "reverse_sensitivity": reverse_sensitivity.tolist(),
        "forward_credit": forward_credit.item(),
        "reverse_credit": reverse_credit.item(),
        "naive_retained_displacement_credit": naive_credit.item(),
        "naive_minus_exact_credit": (naive_credit - forward_credit).item(),
        "forward_reverse_credit_absolute_error": (forward_credit - reverse_credit).abs().item(),
        "finite_differences": finite_differences,
        "forward_transport_hessian_vector_products": len(losses) - 1,
        "trajectory": records,
    }


def _independent_gains(initial, origin_loss, coupled_loss, evaluation_loss):
    """Two independent gains require two distinguishable sensitivity columns."""
    delta = -grad(origin_loss)(initial)

    def trajectory(gains):
        theta = initial + gains[0].exp() * delta
        return theta - 0.5 * gains[1].exp() * grad(coupled_loss)(theta)

    gains = initial.new_zeros(2)
    jacobian = jacrev(trajectory)(gains)
    expected_jacobian = initial.new_tensor([[0.5, -0.5], [-1.0, -0.5]])
    _assert_close(jacobian, expected_jacobian, "independent gain sensitivity columns")
    evaluation_gradient = grad(evaluation_loss)(trajectory(gains))
    credits = -evaluation_gradient @ jacobian
    reverse_credits = grad(lambda a: -evaluation_loss(trajectory(a)))(gains)
    _assert_close(credits, reverse_credits, "independent gain reverse credits")
    _assert_close(credits, initial.new_tensor([35 / 64, 25 / 64]), "analytic independent credits")
    finite_credits = []
    for index in range(2):
        perturbation = torch.zeros_like(gains)
        perturbation[index] = 1e-5
        finite_credit = (evaluation_loss(trajectory(gains - perturbation))
                         - evaluation_loss(trajectory(gains + perturbation))) / 2e-5
        _assert_close(finite_credit, credits[index], f"independent gain {index} finite difference",
                      rtol=2e-6, atol=2e-8)
        finite_credits.append(finite_credit.item())
    determinant = torch.linalg.det(jacobian)
    _assert_close(determinant, initial.new_tensor(-0.75), "independent sensitivity rank")
    summed_trace = jacobian.sum(dim=1)
    return {
        "gain_order": ["origin_log_gain", "subsequent_log_gain"],
        "sensitivity_columns": jacobian.tolist(),
        "sensitivity_determinant": determinant.item(),
        "individual_utility_derivatives": credits.tolist(),
        "reverse_utility_derivatives": reverse_credits.tolist(),
        "finite_difference_utility_derivatives": finite_credits,
        "finite_difference_epsilon": 1e-5,
        "summed_trace": summed_trace.tolist(),
        "summed_trace_credit": (-evaluation_gradient @ summed_trace).item(),
        "interpretation": (
            "The two sensitivity columns are linearly independent. Their sum credits a shared "
            "gain perturbation, not either independent historical gain. A single parameter-sized "
            "forward trace cannot recover arbitrary independent historical credits. One reverse "
            "adjoint can recover all credits for a fixed terminal loss, but only by revisiting "
            "the ordered trajectory and its individual update injections."
        ),
    }


def run_audits() -> dict:
    """Run deterministic CUDA-float64 derivations; raise on a broken identity."""
    if not torch.cuda.is_available():
        raise RuntimeError("Historical credit derivation audit requires CUDA; no CPU fallback")
    device = torch.device("cuda")
    initial = torch.tensor([0.0, 0.5], device=device, dtype=torch.float64)

    def origin_loss(theta):
        return 0.5 * (theta[0] - 1).square()

    def common_loss(theta):
        return 0.5 * (theta[1].square() - 0.75).square()

    def rare_return_loss(theta):
        return 0.5 * theta[0].pow(4)

    def rare_evaluation(theta):
        return 0.5 * (theta[0] - 1).square()

    gap = 17
    rare = _audit_case(
        "rare_feature_gap", initial,
        [origin_loss] + [common_loss] * gap + [rare_return_loss],
        [0.5] + [0.125] * gap + [0.25], rare_evaluation,
        ["rare_origin"] + ["common_only"] * gap + ["rare_return"],
    )
    # The rare feature is absent, not forgotten: the Hessian annihilates its
    # sensitivity throughout the gap. At recurrence the nonlinear Hessian acts.
    for record in rare["trajectory"][1:-1]:
        assert record["sensitivity_after"] == [0.5, 0.0], "gap erased rare sensitivity"
        assert record["hessian_vector"] == [0.0, 0.0], "common-only update mixed rare coordinates"
        assert record["theta_after"][0] == 0.5, "rare parameter moved without its feature"
    assert math.isclose(rare["forward_credit"], 45 / 256, abs_tol=2e-11)
    assert math.isclose(rare["naive_retained_displacement_credit"], 9 / 32, abs_tol=2e-11)
    assert math.isclose(rare["finite_historical_utility"], 175 / 512, abs_tol=2e-11)
    rare["absent_feature_updates"] = gap
    rare["exact_to_naive_credit_ratio"] = rare["forward_credit"] / rare["naive_retained_displacement_credit"]
    rare["analytic_derivation"] = (
        "Origin: r=1/2 and J_r=1/2. Common losses depend only on c, so any number "
        "of gap updates leave both unchanged. At rare return L=r^4/2, H_rr=6r^2=3/2 "
        "and eta=1/4: r'=7/16, J_r'=(1-3/8)/2=5/16. Evaluation gradient is "
        "r'-1=-9/16. Exact credit=45/256; retained-displacement credit=9/32. "
        "The chosen gap is an input sequence, not a credit window or truncation rule."
    )

    coupled_initial = torch.tensor([0.0, 1.0], device=device, dtype=torch.float64)

    def coupled_loss(theta):
        return 0.5 * (theta[0] * theta[1]).square()

    def coupled_evaluation(theta):
        return 0.5 * (theta[1] + 0.25 * theta[0]).square()

    coupled = _audit_case(
        "representation_coupled_delayed_payoff", coupled_initial,
        [origin_loss, coupled_loss], [1.0, 0.5], coupled_evaluation,
        ["representation_origin", "coupled_readout_adaptation"],
    )
    assert math.isclose(coupled["forward_credit"], 35 / 64, abs_tol=2e-11)
    assert math.isclose(coupled["naive_retained_displacement_credit"], -5 / 32, abs_tol=2e-11)
    assert math.isclose(coupled["immediate_utility_derivative"], -5 / 16, abs_tol=2e-11)
    assert math.isclose(coupled["finite_historical_utility"], 39 / 128, abs_tol=2e-11)
    assert coupled["immediate_finite_utility"] < 0 < coupled["finite_historical_utility"]
    assert coupled["naive_retained_displacement_credit"] < 0 < coupled["forward_credit"]
    coupled["analytic_derivation"] = (
        "Parameters (r,w) start at (0,1). Origin L=(r-1)^2/2, eta=1, gives "
        "theta_1=(1,1), delta=J_1=(1,0). Coupled L=(rw)^2/2 has H=[[1,2],[2,1]] "
        "at theta_1. With eta=1/2, theta_2=(1/2,1/2) and J_2=(1/2,-1). "
        "This eta is below 2/lambda_max(H)=2/3; the example is not an exploding "
        "positive-curvature step. Evaluation L=(w+r/4)^2/2 has gradient (5/32,5/8), "
        "so exact historical credit=35/64 but retained-delta credit=-5/32. "
        "Immediate finite utility=-9/32 becomes delayed finite utility=39/128: "
        "the origin lets later learning reduce w. Nonlinear coupling, not a sign "
        "convention, reverses the local attribution."
    )

    return {
        "audit": "utility_credit_audit_v1",
        "device": str(initial.device),
        "dtype": str(initial.dtype),
        "passed": True,
        "evidence_kind": "exact deterministic derivation, not training or optimizer-performance evidence",
        "credit_definition": "d[L_eval(theta_T_without_origin)-L_eval(theta_T(a))]/da at a=0",
        "sensitivity_recurrence": "J_0=0; J_1=exp(a)*delta_0; J_{t+1}=(I-eta_t*H_t)*J_t for t>=1",
        "replay_contract": (
            "Each perturbation replays the complete prescribed example/target sequence. "
            "Losses are pure functions with no randomness, mutable buffers, dropout, "
            "momentum, or data-dependent scheduling. The originating delta is held fixed; "
            "all later gradients are recomputed at their perturbed parameter states."
        ),
        "cost_and_state": {
            "one_origin_forward": (
                "One parameter-sized sensitivity plus current parameter state; one exact "
                "Hessian-vector product per subsequent update in addition to the gradient. "
                "torch.func.jvp(grad(loss)) supplies the gradient and HVP without a dense Hessian. "
                "Streaming forward transport needs no old parameter snapshots for that origin."
            ),
            "independent_origins_forward": (
                "K independent historical gains require K distinguishable sensitivity columns "
                "(O(P*K) storage and generally K HVP directions per later step); a single "
                "summed trace only credits a tied gain. Compression needs assumptions and "
                "is not exact in general."
            ),
            "reverse_credit": (
                "For one terminal objective, reverse transport uses one adjoint and one HVP "
                "per reversed update, with inner products against each origin injection. "
                "It must retain or exactly reconstruct parameter states and replay each "
                "example, target, stochastic realization, buffer and optimizer state. "
                "Reverse AD in this audit retains the complete trajectory."
            ),
            "stateful_updates": (
                "For momentum, adaptive optimizers or differentiable buffers, the state "
                "must include those quantities and transport the full update-map Jacobian. "
                "The plain-SGD Hessian recurrence alone is not exact for those systems."
            ),
        },
        "limitations": [
            "Constructed two-parameter examples prove identities and counterexamples, not learnability or practical scaling.",
            "Finite-difference agreement is numerical evidence at the reported epsilon values, not symbolic proof.",
            "Only deterministic pure-loss plain-SGD trajectories are exercised; stateful/stochastic replay is stated, not tested.",
            "Exact historical credit is not an online forecast and is not a proposed optimization objective.",
        ],
        "cases": {"rare_feature_gap": rare, "representation_coupled_delayed_payoff": coupled},
        "independent_historical_gains": _independent_gains(
            coupled_initial, origin_loss, coupled_loss, coupled_evaluation),
    }

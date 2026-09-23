import torch

from cleanrl.plasticity.predictive_transport_v2 import PredictiveTransportState


def test_parameter_motion_correction_does_not_confuse_new_label_noise_with_motion():
    outcomes = []
    for target in (-100.0, 100.0):
        weight = torch.tensor([[2.0]], device="cuda")
        state = PredictiveTransportState(weight, control="fixed_write")
        state.previous_weight.fill_(1)
        state.forecast.fill_(1)
        state.trust_logit.fill_(-100)
        state.scale_power.fill_(1)
        state.scale_mass.fill_(1)
        gradient = torch.func.grad(lambda w: 0.5 * (w.sum() - target).square())
        torch.compile(state.step, fullgraph=True)(weight, gradient(weight), gradient(state.previous_weight))
        outcomes.append(weight.clone())
    # The trusted population forecast moves from gradient 1 at theta=1 to
    # gradient 2 at theta=2. Either noisy label yields the same useful write.
    for outcome in outcomes:
        torch.testing.assert_close(outcome, torch.zeros_like(outcome), rtol=0, atol=1e-6)


def test_pending_write_credit_survives_feature_absence():
    states = []
    weights = []
    for control in ("predictive", "one_step_credit"):
        weight = torch.zeros(1, 2, device="cuda")
        state = PredictiveTransportState(weight, control=control)
        g = torch.tensor([[-1.0, 0.0]], device="cuda")
        state.step(weight, g, g)
        for _ in range(100):
            state.step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
        before = state.write_log_gain.clone()
        state.step(weight, weight.clone(), state.previous_weight.clone())
        states.append(state.write_log_gain - before)
        weights.append(weight)
    assert states[0][0, 0].item() < -1e-3
    assert abs(states[1][0, 0].item()) < 1e-5


def test_write_credit_distinguishes_harmful_curvature_from_accurate_gradient_prediction():
    def loss(theta):
        prediction = torch.stack((theta[0, 0], 10 * theta[0, 1].square()))
        return 0.5 * (prediction - prediction.new_tensor((0.0, 9.95))).square().sum()

    original = torch.ones(1, 2, device="cuda")
    weight = torch.full_like(original, 0.98)
    assert loss(weight).item() > loss(original).item()
    state = PredictiveTransportState(weight)
    state.credit_anchor.copy_(original)
    gradient = torch.func.grad(loss)(weight)
    state.step(weight, gradient, gradient)
    # The first coordinate's pending write helped; the second's hurt. Exact
    # gradient prediction alone would not distinguish their finite-step utility.
    assert state.write_log_gain[0, 0].item() > 0
    assert state.write_log_gain[0, 1].item() < 0


def test_compiled_repeated_writes_match_eager_future_loss_credit():
    from cleanrl.plasticity.predictive_transport_benchmark_v2 import Args, Runner, configurations

    configs, groups = configurations([{"kind": "signal"}, {"kind": "pure_noise"}])
    eager = Runner(Args(task="sparse"), 16, [{}, {}], 20000, 0, configs, groups)
    captured = Runner(Args(task="sparse"), 16, [{}, {}], 20000, 0, configs, groups)
    captured.capture()
    generator = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn((100, 16), generator=generator, device="cuda")
    targets = torch.randn((100, 2), generator=generator, device="cuda")
    clean = torch.zeros_like(targets)
    eager_step = eager.eager_step
    for index in range(100):
        eager_step(x[index], targets[index], clean[index])
    for buffer, value in zip(captured.inputs, (x, targets, clean)):
        buffer.copy_(value)
    captured.graph.replay()
    # A lost pre-write snapshot makes transport and block utility see zero
    # movement. That changes future predictions, not merely diagnostic fields.
    for control, state in eager.controllers.items():
        rows = groups[control]
        torch.testing.assert_close(captured.weight[rows], eager.weight[rows], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(captured.controllers[control].previous_weight,
                                   state.previous_weight, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(captured.future_write_loss_change[rows],
                                   eager.future_write_loss_change[rows], rtol=1e-4, atol=1e-4)


def test_nonlinear_autograd_preserves_prewrite_snapshot_and_next_prediction():
    from cleanrl.plasticity.predictive_transport_nonlinear_v2 import (
        Args, Experiment, PARAMETERS, batched_forward, draw_inputs,
    )
    from cleanrl.plasticity.predictive_transport_benchmark_v2 import configurations

    generator = torch.Generator(device="cuda").manual_seed(1)
    initial = torch.randn(PARAMETERS, generator=generator, device="cuda") / 32 ** 0.5
    reference = torch.full_like(initial, 1 / 32 ** 0.5)
    configs, groups = configurations([{"kind": "signal"}, {"kind": "pure_noise"}])
    eager = Experiment(Args(), initial, reference, configs, groups)
    captured = Experiment(Args(), initial, reference, configs, groups)
    captured.capture()
    x = draw_inputs(101, generator)
    noise = 5 ** 0.5 * torch.randn(100, generator=generator, device="cuda")
    clean = torch.stack((x[:100, 0] * x[:100, 1], torch.zeros_like(noise)), -1)
    targets = clean + noise[:, None]
    for index in range(100):
        # Compare each transition from identical state. Independent 100-step
        # nonlinear trajectories amplify legitimate FP32 rounding differences.
        for destination, source in zip(eager.states, captured.states):
            destination.copy_(source)
        before = captured.weight.clone()
        eager.eager_step(x[index], targets[index], clean[index])
        captured.compiled(x[index], targets[index], clean[index])
        for control, state in captured.controllers.items():
            rows = groups[control]
            torch.testing.assert_close(state.previous_weight, before[rows], rtol=0, atol=0)
            torch.testing.assert_close(batched_forward(captured.weight[rows], x[index + 1]),
                                       batched_forward(eager.weight[rows], x[index + 1]),
                                       rtol=1e-3, atol=1e-4)

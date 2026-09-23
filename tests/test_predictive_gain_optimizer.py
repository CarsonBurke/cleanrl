import torch

from cleanrl.plasticity.predictive_gain_optimizer_v1 import PredictiveGainState


def test_unit_initial_gain_can_learn_one_clean_observation_completely():
    weight = torch.zeros(1, 1, device="cuda")
    state = PredictiveGainState(weight)
    state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.ones_like(weight))
    torch.testing.assert_close(state.log_gain, torch.zeros_like(weight))


def test_later_helpful_and_harmful_credit_move_gain_in_opposite_directions():
    weights = [torch.zeros(1, 1, device="cuda") for _ in range(2)]
    states = [PredictiveGainState(weight) for weight in weights]
    for state, weight in zip(states, weights):
        state.step(weight, torch.full_like(weight, -0.1), torch.full_like(weight, 0.1))
    states[0].step(weights[0], torch.full_like(weights[0], -0.1), torch.full_like(weights[0], 0.1))
    states[1].step(weights[1], torch.full_like(weights[1], 0.1), torch.full_like(weights[1], 0.1))
    torch.testing.assert_close(states[0].log_gain, torch.full_like(weights[0], 0.1))
    torch.testing.assert_close(states[1].log_gain, torch.full_like(weights[1], -0.1))


def test_sample_stability_does_not_tax_an_absent_coordinate():
    weight = torch.zeros(1, 2, device="cuda")
    state = PredictiveGainState(weight)
    state.step(weight, torch.tensor([[-2.0, 0.0]], device="cuda"),
               torch.tensor([[4.0, 0.0]], device="cuda"))
    torch.testing.assert_close(weight, torch.tensor([[0.5, 0.0]], device="cuda"))
    torch.testing.assert_close(state.log_gain[:, 1], torch.zeros(1, device="cuda"), rtol=0, atol=0)


def test_sign_credit_reopens_despite_large_old_magnitude_scale():
    weight = torch.zeros(1, 1, device="cuda")
    state = PredictiveGainState(weight, control="sign")
    state.log_gain.fill_(-30)
    state.credit_scale.fill_(1e10)
    state.trace.fill_(1e-12)
    state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    torch.testing.assert_close(state.log_gain, torch.full_like(weight, -29.9))


def test_compiled_zero_warmup_and_credit_relabelling():
    weights = [torch.zeros(1, 2, device="cuda") for _ in range(2)]
    states = [PredictiveGainState(weight, control=control)
              for weight, control in zip(weights, ("predictive", "shuffled"))]
    steps = [torch.compile(state.step, fullgraph=True) for state in states]
    for step, weight in zip(steps, weights):
        step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
        torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
        step(weight, torch.tensor([[-0.1, 0.0]], device="cuda"), torch.full_like(weight, 0.1))
        step(weight, torch.tensor([[0.0, -0.1]], device="cuda"), torch.full_like(weight, 0.1))
    torch.testing.assert_close(states[0].log_gain, torch.zeros_like(weights[0]), rtol=0, atol=0)
    torch.testing.assert_close(states[1].log_gain, torch.tensor([[0.0, 0.1]], device="cuda"))

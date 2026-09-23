"""Numerical contracts for the redesign; all GPU execution belongs in mlq."""
import math

import torch

from cleanrl.plasticity.contextual_belief_v2 import ContextualBeliefState
from cleanrl.shared.runtime import configure_runtime


def test_gaussian_conditioning_matches_information_form_without_covariance_clipping():
    dtype, device = torch.float64, "cuda"
    mean = torch.tensor([[[1., -2., .5]]], dtype=dtype, device=device)
    factor = torch.tensor([[2.,0.,0.],[.3,1.,0.],[-.2,.1,.5]], dtype=dtype, device=device)
    covariance = (factor @ factor.T)[None, None]
    phi = torch.tensor([[[1.,.7,-.2]]], dtype=dtype, device=device)
    noise = torch.tensor([[.4]], dtype=dtype, device=device)
    residual = torch.tensor([[1.3]], dtype=dtype, device=device)
    observed = (mean * phi).sum(-1) + residual
    posterior_mean, posterior_covariance = ContextualBeliefState.posterior(mean,covariance,phi,residual,noise)
    inverse = torch.linalg.inv(covariance)
    expected_covariance = torch.linalg.inv(inverse + phi[..., :, None] * phi[..., None, :] / noise[..., None, None])
    information = (inverse @ mean[..., None]).squeeze(-1) + phi * observed[..., None] / noise[..., None]
    expected_mean = (expected_covariance @ information[..., None]).squeeze(-1)
    torch.testing.assert_close(posterior_covariance,expected_covariance,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(posterior_mean,expected_mean,rtol=1e-12,atol=1e-12)
    assert bool((torch.linalg.eigvalsh(posterior_covariance) > 0).all())


def test_noise_update_uses_mixture_score_including_epistemic_uncertainty():
    state = ContextualBeliefState(torch.zeros((1,2),device="cuda"))
    state.log_variance.requires_grad_()
    context = torch.tensor([[[.2,-.5],[-.7,.1]]],device="cuda")
    residual = torch.tensor([3.],device="cuda")
    _, updated, metrics = state.belief_transition(context,residual,torch.ones((1,2),dtype=torch.bool,device="cuda"))
    score = torch.autograd.grad(-metrics[0].sum(),state.log_variance)[0]
    torch.testing.assert_close(updated["variance_accumulator"],score.square(),rtol=1e-11,atol=1e-12)
    expected = state.log_variance + .1 * score / (score.square()+1e-12).sqrt()
    torch.testing.assert_close(updated["log_variance"],expected,rtol=1e-11,atol=1e-12)


@torch.no_grad()
def test_prior_gain_distinguishes_calibrated_unknown_and_biased_predictions():
    gains = []
    for mean, variance in ((0.,.001),(0.,9.),(3.,.001)):
        state = ContextualBeliefState(torch.zeros((1,1),device="cuda"),control="no_change")
        state.log_odds.fill_(20)
        state.mean[...,0].fill_(mean)
        state.covariance.mul_(variance)
        state.log_variance[...,0].fill_(math.log(9))
        gains.append(float(state.predictive(torch.zeros((1,1,2),device="cuda"))["gain"]))
    assert gains[0] < .001
    assert gains[1] > .49 and gains[2] > .49


@torch.no_grad()
def test_suppressed_write_does_not_suppress_revision_evidence():
    weight = torch.zeros((1,1),device="cuda")
    state = ContextualBeliefState(weight)
    state.log_odds.fill_(-80)
    state.covariance.mul_(1e-6)
    context = torch.zeros((1,1,2),device="cuda")
    before = state.predictive(context)["gain"].clone()
    residual = weight.new_tensor([20.])
    state.step(weight,residual[:,None],torch.ones_like(weight),context,residual)
    after = state.predictive(context)["gain"]
    assert before.item() < .001
    assert after.item() > .9
    assert state.change_probability_sum.item() > .9
    # The surprising observation revises NEXT gain, not its own already-used gain.
    expected = -before * 20
    torch.testing.assert_close(weight.double(),expected,rtol=1e-6,atol=1e-8)


@torch.no_grad()
def test_absence_preserves_belief_and_outlier_cannot_change_its_own_gain():
    outputs = []
    for residual in (1.,10.):
        weight = torch.zeros((1,1),device="cuda")
        state = ContextualBeliefState(weight)
        saved = {key:value.clone() for key,value in state.buffers().items()}
        context = torch.zeros((1,1,2),device="cuda")
        for _ in range(100):
            state.step(weight,torch.zeros_like(weight),torch.zeros_like(weight),context,weight.new_tensor([100.]))
        for key,value in state.buffers().items():
            torch.testing.assert_close(value,saved[key],rtol=0,atol=0)
        state.step(weight,weight.new_tensor([[residual]]),torch.ones_like(weight),context,weight.new_tensor([residual]))
        outputs.append(weight.clone())
    torch.testing.assert_close(outputs[1],10*outputs[0],rtol=1e-6,atol=1e-7)


@torch.no_grad()
def test_linear_cuda_graph_preserves_predictions_and_mixture_calibration():
    from cleanrl.plasticity.contextual_belief_benchmark_v2 import Args,Runner,configurations,CHUNK
    configure_runtime(matmul_precision="highest",allow_tf32=False)
    args = Args(task="sparse")
    streams = [{"kind":"signal","alpha":1.},{"kind":"pure_noise","alpha":0.}]
    configs,groups = configurations(streams)
    eager = Runner(args,16,streams,20000,0,configs,groups)
    compiled = Runner(args,16,streams,20000,0,configs,groups)
    compiled.capture()
    rng = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.rand((CHUNK,16),generator=rng,device="cuda") < .2).float()
    targets = torch.randn((CHUNK,2),generator=rng,device="cuda")
    clean = torch.zeros_like(targets)
    compiled.advance(x,targets,clean)
    for offset in range(CHUNK):
        eager.eager_step(x[offset],targets[offset],clean[offset])
    rows = slice(groups["belief"].start,None)
    torch.testing.assert_close(compiled.weight[rows],eager.weight[rows],rtol=5e-4,atol=2e-5)
    for name in compiled.controllers:
        torch.testing.assert_close(compiled.controllers[name].nll_sum,eager.controllers[name].nll_sum,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(compiled.controllers[name].gain_sum,eager.controllers[name].gain_sum,rtol=1e-5,atol=1e-5)


@torch.no_grad()
def test_nonlinear_compiled_transition_matches_next_example_prediction():
    from cleanrl.plasticity.contextual_belief_nonlinear_v2 import Args,Experiment,PARAMETERS,batched_forward
    from cleanrl.plasticity.contextual_belief_benchmark_v2 import configurations
    configure_runtime(matmul_precision="highest",allow_tf32=False)
    rng = torch.Generator(device="cuda").manual_seed(1)
    initial = torch.randn(PARAMETERS,generator=rng,device="cuda") * .05
    streams = [{"kind":"signal","alpha":1.},{"kind":"pure_noise","alpha":0.}]
    configs,groups = configurations(streams)
    eager = Experiment(Args(),initial,configs,groups)
    compiled = Experiment(Args(),initial,configs,groups)
    rows = slice(groups["belief"].start,None)
    for _ in range(100):
        x = torch.randn(32,generator=rng,device="cuda")
        target = torch.randn(2,generator=rng,device="cuda")
        for a,b in zip(eager.states,compiled.states):
            a.copy_(b)
        eager.eager_step(x,target,torch.zeros_like(target))
        compiled.compiled(x,target,torch.zeros_like(target))
        next_x = torch.randn(32,generator=rng,device="cuda")
        torch.testing.assert_close(batched_forward(compiled.weight[rows],next_x),
                                   batched_forward(eager.weight[rows],next_x),rtol=1e-3,atol=1e-4)
        for name in compiled.controllers:
            torch.testing.assert_close(compiled.controllers[name].nll_sum,eager.controllers[name].nll_sum,rtol=1e-5,atol=1e-5)


@torch.no_grad()
def test_unseen_context_does_not_change_prior_plasticity_by_feature_norm():
    weight = torch.zeros((1, 3), device="cuda")
    state = ContextualBeliefState(weight)
    empty_context = torch.zeros((1, 3, 2), device="cuda")
    varied_context = weight.new_tensor([[[0., 0.], [1., 1.], [.2, -.8]]])
    baseline = state.predictive(empty_context)
    contextual = state.predictive(varied_context)
    for key in ("gain", "predictive_variance"):
        torch.testing.assert_close(contextual[key], baseline[key], rtol=1e-14, atol=1e-14)


@torch.no_grad()
def test_redundant_parameters_do_not_amplify_predictive_confidence():
    corrections = []
    for parameters in (1, 64):
        weight = torch.zeros((1, parameters), device="cuda")
        jacobian = torch.ones_like(weight)
        context = torch.zeros((1, parameters, 2), device="cuda")
        residual = weight.new_tensor([2.])
        state = ContextualBeliefState(weight)
        prior_gain = state.predictive(context)["gain"][:, 0]
        state.step(weight, residual[:, None] * jacobian, jacobian, context, residual)
        correction = (weight * jacobian).sum(1)
        torch.testing.assert_close(correction.double(), -prior_gain * residual, rtol=1e-6, atol=1e-8)
        corrections.append(correction)
    torch.testing.assert_close(corrections[0], corrections[1], rtol=1e-6, atol=1e-8)

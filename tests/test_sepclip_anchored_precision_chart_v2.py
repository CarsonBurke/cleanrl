"""Versioned chart contracts after preserved 11770 failures; CUDA/MLQ only.

Fixed-shape compiled jacrev avoids the observed symbolic Inductor CantSplit.
FP64 references account for frozen softplus's threshold20 approximation and
the tiny-excess inverse series, rather than loosening an identity tolerance.
"""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
import pytest
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as old
from cleanrl import ppo_continuous_action_sepclip_anchored_precision_chart_v13 as new
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_only():
    assert torch.cuda.is_available(), 'CUDA required; execute via mlq'
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def compile_fixed(fn, **kwargs):
    return torch.compile(fn, dynamic=False, **kwargs)


def exact_inverse_reference(excess):
    # Independent positive-expm1 identity below1; no cancellation there.
    # Guard unused expressions so the reference does not overflow at large y.
    small = torch.log(torch.expm1(excess.clamp(max=1)))
    large = excess + torch.log1p(-torch.exp(-excess.clamp(min=1)))
    return torch.where(excess < 1, small, large)


def fp64_budget(scale):
    # Two stable paths, each budgeted for <=128 rounding/primitive errors.
    # This is an explicit engineering error model, not a libdevice proof.
    # Primitive exp/log accuracy and compiled reassociation must be validated
    # by the queued measurement; eps is NOT used to cover threshold bias.
    unit = torch.finfo(torch.float64).eps / 2
    gamma = 128 * unit / (1 - 128 * unit)
    return 2 * gamma * (1 + scale.abs())


def frozen_shape_expectation(excess, decoded):
    """Independent analytic shape/chain target and approximation bounds.

    For decoded>20, frozen F.softplus returns decoded rather than exact
    log(1+exp(decoded)). Exact inverse z=log(expm1(y)) then has deficit
    -log(1-exp(-y)) and dshape/dlog(y)=y/(1-exp(-y)).
    For y<exp(-10), the inverse series overestimates z by at most y^4/2880;
    its log-y derivative error is at most y^4/720 (Euler sinh product).
    """
    linear = decoded > 20
    v = torch.where(linear, torch.exp(-excess), torch.zeros_like(excess))
    deficit = -torch.log1p(-v)
    chain = excess + excess * v / (1 - v)
    small = excess < np.exp(-10.)
    inverse_error = torch.where(small, excess.pow(4) / 2880, torch.zeros_like(excess))
    inverse_chain_error = torch.where(small, excess.pow(4) / 720, torch.zeros_like(excess))
    exact_inverse_chain = excess / (-torch.expm1(-excess))
    # Softplus is1-Lipschitz; sigmoid is1/4-Lipschitz.
    shape_bound = inverse_error
    chain_bound = inverse_chain_error + exact_inverse_chain * inverse_error / 4
    return 1 + excess - deficit, chain, shape_bound, chain_bound, deficit


def pair(chart):
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    descriptor=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.,1.,(6,),dtype=np.float32))
    torch.manual_seed(1);a=old.Agent(descriptor).cuda();a.normalize_matrices()
    torch.manual_seed(1);b=new.Agent(descriptor,policy_chart=chart).cuda();b.normalize_matrices();b.actor.freeze_initial_readout()
    return a,b


def test_alpha_beta_control_is_exact_original_function_and_parameter_gradient():
    a,b=pair('alpha_beta')
    x=torch.randn(32,17,device='cuda')
    forward_a=compile_fixed(a.actor,fullgraph=True,options={'triton.cudagraphs':False})
    forward_b=compile_fixed(b.actor,fullgraph=True,options={'triton.cudagraphs':False})
    y,z=forward_a(x),forward_b(x)
    torch.testing.assert_close(y,z,atol=0,rtol=0)
    ga=torch.autograd.grad(y.square().mean(),tuple(a.actor.parameters()))
    gb=torch.autograd.grad(z.square().mean(),tuple(b.actor.parameters()))
    for p,q in zip(ga,gb):torch.testing.assert_close(p,q,atol=0,rtol=0)
    # Control remains the original function after any independent head update.
    with torch.no_grad():
        for p,q in zip(a.actor.parameters(),b.actor.parameters()):
            delta=torch.randn_like(p)*.1;p.add_(delta);q.add_(delta)
    torch.testing.assert_close(forward_a(x),forward_b(x),atol=0,rtol=0)


def test_initial_chart_preserves_policy_hidden_jacobian_and_full_head_flow():
    a,b=pair('mode_log_precision')
    x=torch.randn(4,17,device='cuda')
    fa=compile_fixed(a.actor,fullgraph=True,options={'triton.cudagraphs':False})
    fb=compile_fixed(b.actor,fullgraph=True,options={'triton.cudagraphs':False})
    ya,yb=fa(x),fb(x)
    torch.testing.assert_close(1+torch.nn.functional.softplus(ya),1+torch.nn.functional.softplus(yb),rtol=2e-6,atol=2e-6)
    ja=compile_fixed(torch.func.jacrev(a.actor),fullgraph=True,options={'triton.cudagraphs':False})(x)
    jb=compile_fixed(torch.func.jacrev(b.actor),fullgraph=True,options={'triton.cudagraphs':False})(x)
    torch.testing.assert_close(ja,jb,rtol=3e-5,atol=2e-5)
    heads=tuple(b.actor.head.parameters())+(b.actor.readout_gain,)
    # Independent compiled forward, without retaining donated backward buffers.
    weights=torch.arange(1,13,device='cuda').float()
    gradients=torch.autograd.grad((fb(x)*weights).sum(),heads)
    assert all(float(g.norm())>0 for g in gradients)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())


def test_host_mirror_matches_chart_after_updates_without_random_draws():
    _,b=pair('mode_log_precision')
    with torch.no_grad():
        b.actor.head[0].bias.add_(torch.linspace(-2,2,12,device='cuda'))
        b.actor.readout_gain.mul_(1.7)
    x=np.random.default_rng(1).normal(size=(16,17)).astype(np.float32)
    state=torch.get_rng_state().clone();cuda_state=torch.cuda.get_rng_state().clone()
    mirror=new.ResidualHostMirror(b.actor,16)
    torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0)
    torch.testing.assert_close(cuda_state,torch.cuda.get_rng_state(),atol=0,rtol=0)
    forward=compile_fixed(b.actor,fullgraph=True,options={'triton.cudagraphs':False})
    expected=forward(torch.as_tensor(x,device='cuda')).detach().cpu().numpy()
    actual=mirror(x).copy()
    np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=3e-5)


def test_chart_shape_jacobian_and_wide_numerical_range():
    configure_runtime()
    anchor=torch.linspace(-8,8,48,device='cuda',dtype=torch.float64).reshape(4,12)
    trainable=(anchor+torch.linspace(-4,4,48,device='cuda',dtype=torch.float64).reshape(4,12)).requires_grad_()
    f=compile_fixed(new.decode_chart,fullgraph=True,options={'triton.cudagraphs':False})
    output=f(anchor,trainable)
    alpha,beta=(1+torch.nn.functional.softplus(output)).chunk(2,-1)
    za,zb=anchor.chunk(2,-1);dm,dl=(trainable-anchor).chunk(2,-1)
    # Alternative exact-softplus and multiplicative precision reference.
    aa=torch.logaddexp(torch.zeros_like(za),za)
    bb=torch.logaddexp(torch.zeros_like(zb),zb)
    r=aa.log()-bb.log()+dm
    assert bool((r.abs()<20).all()), 'Fixture isolates outer softplus threshold'
    mode=torch.sigmoid(r)
    precision=(aa+bb)*dl.exp()
    excess_a,excess_b=precision*mode,precision*(1-mode)
    out_a,out_b=output.chunk(2,-1)
    expected_a,chain_a,bound_a,chain_bound_a,_=frozen_shape_expectation(excess_a,out_a)
    expected_b,chain_b,bound_b,chain_bound_b,_=frozen_shape_expectation(excess_b,out_b)
    budget=fp64_budget(precision)
    assert bool(((alpha-expected_a).abs()<=bound_a+budget).all())
    assert bool(((beta-expected_b).abs()<=bound_b+budget).all())
    grad=torch.autograd.grad(alpha.sum(),trainable)[0]
    gm,gc=grad.chunk(2,-1)
    assert bool(((gm-chain_a*(1-mode)).abs()<=chain_bound_a+budget).all())
    assert bool(((gc-chain_a).abs()<=chain_bound_a+budget).all())
    # A fresh graph avoids retaining/donating the first compiled backward.
    beta_again=(1+torch.nn.functional.softplus(f(anchor,trainable))).chunk(2,-1)[1]
    beta_gradient=torch.autograd.grad(beta_again.sum(),trainable)[0]
    bm,bc=beta_gradient.chunk(2,-1)
    assert bool(((bm+chain_b*mode).abs()<=chain_bound_b+budget).all())
    assert bool(((bc-chain_b).abs()<=chain_bound_b+budget).all())
    assert torch.isfinite(output).all()


def test_inverse_series_and_outer_softplus_threshold_have_derived_error_budgets():
    y=torch.tensor([np.exp(-14),np.exp(-10.001),np.exp(-10),np.exp(-9.999),
                    .1,1.,19.99,20.01,23.862,40.],device='cuda',dtype=torch.float64)
    log_y=y.log().detach().requires_grad_()
    decode=compile_fixed(new.inverse_softplus_from_log,fullgraph=True,
                         options={'triton.cudagraphs':False})
    decoded=decode(log_y)
    actual_y=log_y.detach().exp()
    exact=exact_inverse_reference(actual_y)
    tiny=actual_y<np.exp(-10.)
    inverse_bound=torch.where(tiny,actual_y.pow(4)/2880,torch.zeros_like(actual_y))
    numeric=fp64_budget(exact)
    difference=decoded-exact
    assert bool((difference>=-numeric).all())
    assert bool((difference<=inverse_bound+numeric).all())
    shapes=1+torch.nn.functional.softplus(decoded)
    expected,chain,shape_bound,chain_bound,deficit=frozen_shape_expectation(actual_y,decoded)
    derivative=torch.autograd.grad(shapes.sum(),log_y)[0]
    assert bool(((shapes-expected).abs()<=shape_bound+fp64_budget(actual_y)).all())
    assert bool(((derivative-chain).abs()<=chain_bound+fp64_budget(actual_y)).all())
    linear=decoded>20
    assert bool(linear[7:].all()) and not bool(linear[:7].any())
    v=torch.exp(-actual_y[linear])
    assert bool((deficit[linear]>=v-fp64_budget(v)).all())
    assert bool((deficit[linear]<=v/(1-v)+fp64_budget(v)).all())
    # At20.01 the missing exact-softplus term exceeds ordinary FP64 roundoff.
    assert float(deficit[7])>100*float(fp64_budget(actual_y)[7])
    assert float((1+actual_y[7]-shapes[7]))>50*float(fp64_budget(actual_y)[7])


def test_static_compiled_initial_input_jacobian_has_independent_centered_primal_reference():
    a,b=pair('mode_log_precision')
    x=torch.linspace(-.9,1.1,4*17,device='cuda').reshape(4,17)
    direction=torch.linspace(.7,-.6,4*17,device='cuda').reshape(4,17)
    for agent in (a,b):
        jacobian=compile_fixed(torch.func.jacrev(agent.actor),fullgraph=True,
                              options={'triton.cudagraphs':False})(x).clone()
        response=torch.einsum('soij,ij->so',jacobian,direction).double()
        primal=compile_fixed(agent.actor,fullgraph=True,options={'triton.cudagraphs':False})
        assert float(response.norm())>1e-6
        for h in (.02,.01,.005):
            plus=primal(x+h*direction).detach().clone().double()
            minus=primal(x-h*direction).detach().clone().double()
            finite=(plus-minus)/(2*h)
            relative=float((finite-response).norm()/response.norm())
            cosine=float((finite.flatten()@response.flatten())/(finite.norm()*response.norm()))
            assert relative<.02 and cosine>.999


def test_initial_hidden_pg_preserved_conditioned_on_actor_clip_factor():
    a,b=pair('mode_log_precision')
    obs=torch.randn(64,17,device='cuda');native=torch.rand(64,6,device='cuda')*.8+.1
    advantage=torch.linspace(-.02,.02,64,device='cuda')
    def objective(agent,x,n,adv):
        alpha,beta,_=agent.get_policy_and_value(x)
        return -(agent.action_logprob(alpha,beta,n)*adv).mean()
    losses=[compile_fixed(lambda x,n,adv,agent=agent:objective(agent,x,n,adv),fullgraph=True,
        options={'triton.cudagraphs':False}) for agent in (a,b)]
    norms=[];gradients=[]
    for agent,fn in zip((a,b),losses):
        fn(obs,native,advantage).backward()
        norms.append(torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()),.5,foreach=True))
        gradients.append({k:p.grad.detach().clone() for k,p in agent.actor.named_parameters()})
    assert all(float(n)<.5 for n in norms),'Fixture must identify the no-clipping initial geometry'
    for key in gradients[0]:
        if key.startswith(('first.','second.')):
            torch.testing.assert_close(gradients[0][key],gradients[1][key],rtol=3e-5,atol=3e-8)
    assert not torch.allclose(gradients[0]['head.0.weight'],gradients[1]['head.0.weight'],rtol=1e-3,atol=1e-7)


def test_fp32_log_excess_inverse_preserves_tiny_shape_domain_and_gradient():
    configure_runtime()
    values=torch.tensor([-200.,-120.,-100.,-80.,-20.,-10.001,-10.,-9.999,0.,5.,20.],device='cuda',requires_grad=True)
    fn=compile_fixed(new.inverse_softplus_from_log,fullgraph=True,options={'triton.cudagraphs':False})
    logits=fn(values)
    frozen=logits.detach().clone()
    gradient=torch.autograd.grad(logits.sum(),values)[0]
    assert bool(torch.isfinite(frozen).all()) and bool(torch.isfinite(gradient).all())
    torch.testing.assert_close(gradient[:4],torch.ones_like(gradient[:4]),rtol=1e-6,atol=1e-6)
    host=new.inverse_softplus_from_log_host(values.detach().cpu().numpy())
    np.testing.assert_allclose(host,frozen.cpu().numpy(),rtol=2e-6,atol=2e-6)
    # Compare equivalent shapes in the actual FP32 policy law, not inverse
    # logits against an underflowing direct reference.
    torch.testing.assert_close(1+torch.nn.functional.softplus(frozen),1+values.detach().exp(),rtol=2e-6,atol=2e-6)


def test_saved_policy_evaluation_uses_trained_chart_frozen_rows_and_staged_actions(monkeypatch):
    _,agent=pair('mode_log_precision')
    norm=SimpleNamespace(means=np.stack([np.zeros(17),np.ones(17)*2]),
        variances=np.stack([np.ones(17),np.ones(17)*4]),epsilon=1e-8,clip=10.)
    raw=np.stack([np.arange(17),np.arange(17)*2]).astype(np.float32)
    expected_obs=np.clip((raw-norm.means)/np.sqrt(norm.variances+norm.epsilon),-10,10).astype(np.float32)
    expected=compile_fixed(agent.actor,fullgraph=True,options={'triton.cudagraphs':False})(
        torch.as_tensor(expected_obs,device='cuda')).detach().cpu().numpy()
    physical=np.zeros((2,6),np.float32)
    class EpisodeEnv:
        closed=False
        def reset(self,seed):
            assert seed==101
            return raw.copy(),{}
        def step(self,actions):
            assert not np.shares_memory(actions,physical)
            return raw.copy(),np.array([3.,4.]),np.zeros(2,bool),np.ones(2,bool),{}
        def close(self):self.closed=True
    env=EpisodeEnv()
    monkeypatch.setattr(new,'make_mujoco_vector_env',lambda *a,**k:env)
    def sampler(logits,rng):
        np.testing.assert_allclose(logits,expected,rtol=3e-5,atol=3e-5)
        return np.zeros_like(physical),physical
    monkeypatch.setattr(new,'make_beta_sampler',lambda *a,**k:sampler)
    saved={k:getattr(norm,k).copy() for k in ('means','variances')}
    args=SimpleNamespace(seed=1,num_envs=2,env_threads=2,env_id='HalfCheetah-v4',capture_video=False)
    assert new.evaluate_frozen_policy(agent,norm,args,'contract',episodes=2)==[3.,4.]
    assert env.closed
    for key,value in saved.items():np.testing.assert_array_equal(getattr(norm,key),value)

"""Fresh chart preserves initialization/hidden sensitivity; queued CUDA contracts."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as old
from cleanrl import ppo_continuous_action_sepclip_anchored_precision_chart_v13 as new
from cleanrl.shared.runtime import configure_runtime


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
    forward_a=torch.compile(a.actor,fullgraph=True,options={'triton.cudagraphs':False})
    forward_b=torch.compile(b.actor,fullgraph=True,options={'triton.cudagraphs':False})
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
    fa=torch.compile(a.actor,fullgraph=True,options={'triton.cudagraphs':False})
    fb=torch.compile(b.actor,fullgraph=True,options={'triton.cudagraphs':False})
    ya,yb=fa(x),fb(x)
    torch.testing.assert_close(1+torch.nn.functional.softplus(ya),1+torch.nn.functional.softplus(yb),rtol=2e-6,atol=2e-6)
    ja=torch.compile(torch.func.jacrev(a.actor),fullgraph=True,options={'triton.cudagraphs':False})(x)
    jb=torch.compile(torch.func.jacrev(b.actor),fullgraph=True,options={'triton.cudagraphs':False})(x)
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
    forward=torch.compile(b.actor,fullgraph=True,options={'triton.cudagraphs':False})
    expected=forward(torch.as_tensor(x,device='cuda')).detach().cpu().numpy()
    actual=mirror(x).copy()
    np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=3e-5)


def test_chart_shape_jacobian_and_wide_numerical_range():
    configure_runtime()
    anchor=torch.linspace(-8,8,48,device='cuda',dtype=torch.float64).reshape(4,12)
    trainable=(anchor+torch.linspace(-4,4,48,device='cuda',dtype=torch.float64).reshape(4,12)).requires_grad_()
    f=torch.compile(new.decode_chart,fullgraph=True,options={'triton.cudagraphs':False})
    output=f(anchor,trainable)
    alpha,beta=(1+torch.nn.functional.softplus(output)).chunk(2,-1)
    za,zb=anchor.chunk(2,-1);dm,dl=(trainable-anchor).chunk(2,-1)
    mode=torch.sigmoid(torch.nn.functional.softplus(za).log()-torch.nn.functional.softplus(zb).log()+dm)
    precision=(torch.logaddexp(torch.nn.functional.softplus(za).log(),torch.nn.functional.softplus(zb).log())+dl).exp()
    torch.testing.assert_close(alpha,1+precision*mode,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(beta,1+precision*(1-mode),rtol=1e-12,atol=1e-12)
    grad=torch.autograd.grad(alpha.sum(),trainable)[0]
    gm,gc=grad.chunk(2,-1)
    torch.testing.assert_close(gm,precision*mode*(1-mode),rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(gc,precision*mode,rtol=1e-10,atol=1e-10)
    assert torch.isfinite(output).all()


def test_initial_hidden_pg_preserved_conditioned_on_actor_clip_factor():
    a,b=pair('mode_log_precision')
    obs=torch.randn(64,17,device='cuda');native=torch.rand(64,6,device='cuda')*.8+.1
    advantage=torch.linspace(-.02,.02,64,device='cuda')
    def objective(agent,x,n,adv):
        alpha,beta,_=agent.get_policy_and_value(x)
        return -(agent.action_logprob(alpha,beta,n)*adv).mean()
    losses=[torch.compile(lambda x,n,adv,agent=agent:objective(agent,x,n,adv),fullgraph=True,
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
    fn=torch.compile(new.inverse_softplus_from_log,fullgraph=True,options={'triton.cudagraphs':False})
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
    expected=torch.compile(agent.actor,fullgraph=True,options={'triton.cudagraphs':False})(
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

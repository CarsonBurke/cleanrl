"""Saved-reference metadata and CUDA/host contracts; queue through mlq."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strong_reference", ROOT/"scripts/audit_sepclip_strong_reference.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope="module")
def saved():
    audit.configure_runtime(matmul_precision="highest", allow_tf32=False)
    assert torch.cuda.is_available(), "Submit through mlq on CUDA"
    assert audit.file_hash(audit.DEFAULT_CHECKPOINT) == audit.EXPECTED_HASH
    snapshot = torch.load(audit.DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    envs = audit.make_mujoco_vector_env("HalfCheetah-v4",16,num_threads=2)
    try:
        agent, metadata = audit.load_agent(snapshot,envs)
        raw, _ = envs.reset(seed=1)
        norm = audit.physics.norm_from_snapshot(snapshot,np.arange(16))
        obs = audit.physics.normalized(norm,raw)
        yield snapshot,agent,envs,obs,metadata
    finally:
        envs.close()


def test_normalizer_is_exact_rows_and_metadata_failures_are_rejected(saved):
    snapshot,_,_,obs,metadata = saved
    assert metadata["rows"] == list(range(16))
    assert obs.dtype == np.float32 and obs.shape == (16,17)
    for name,value in (("epsilon",1e-5),("clip",5.),("means",torch.zeros((1,17),dtype=torch.float64)),
                       ("variances",torch.full((16,17),-1.,dtype=torch.float64)),
                       ("counts",torch.zeros(16,dtype=torch.float64))):
        broken=dict(snapshot,obs_norm=dict(snapshot["obs_norm"]))
        broken["obs_norm"][name]=value
        with pytest.raises(ValueError):
            audit.validate_metadata(broken)
    broken=dict(snapshot,args=dict(snapshot["args"],activation="relu"))
    with pytest.raises(ValueError,match="architecture"):
        audit.validate_metadata(broken)


def test_saved_model_requires_strict_keys_and_physical_bounds(saved):
    snapshot,agent,envs,_,_ = saved
    broken=dict(snapshot,model=dict(snapshot["model"]))
    broken["model"].pop(next(iter(broken["model"])))
    with pytest.raises(RuntimeError,match="Missing key"):
        audit.load_agent(broken,envs)
    original=agent.action_low.clone()
    try:
        agent.action_low.fill_(-.5)
        with pytest.raises(ValueError,match="action_low"):
            audit.validate_bounds(agent,envs)
    finally:
        agent.action_low.copy_(original)


def test_host_matches_compiled_cuda_and_materializes_permanent_buffers(saved):
    _,agent,_,obs,_ = saved
    mirror=audit.make_host_mirror(agent.actor,16)
    cuda=audit.compiled_policy(agent)
    a,b,z,errors=audit.checked_host_policy(mirror,cuda,obs)
    copies=(a.copy(),b.copy(),z.copy())
    shifted=np.clip(obs+np.linspace(-.25,.25,17,dtype=np.float32),-10,10).astype(np.float32)
    audit.checked_host_policy(mirror,cuda,shifted)
    for value,preserved in zip((a,b,z),copies):
        np.testing.assert_array_equal(value,preserved)
    assert max(errors.values()) < 1e-3
    class WrongMirror:
        def __call__(self,x):
            return mirror(x).copy()+.1
    with pytest.raises(ValueError,match="host/CUDA logits mismatch"):
        audit.checked_host_policy(WrongMirror(),cuda,obs)


def test_mean_mode_native_bounds_and_physical_transform(saved):
    _,agent,_,obs,_ = saved
    mirror=audit.make_host_mirror(agent.actor,16)
    a,b,z,_=audit.checked_host_policy(mirror,audit.compiled_policy(agent),obs)
    for mode in ("sample","mean","mode"):
        native=audit.reference_native_action(a,b,mode,np.full(a.shape,.5))
        assert np.isfinite(native).all() and np.all((native>=0)&(native<=1))
        physical=2*native-1
        assert np.all((physical>=-1)&(physical<=1))
    mean=audit.physics.native_action(a,b,"mean",None,z)
    np.testing.assert_array_equal(mean,(a/(a+b)).astype(np.float32))


def test_rounded_shape_one_is_a_valid_beta_law_with_explicit_mode_tie():
    next_one=np.nextafter(np.float32(1),np.float32(2))
    a=np.array([[1.,3.,1.,1.,next_one,next_one]],np.float32)
    b=np.array([[3.,1.,1.,next_one,1.,next_one]],np.float32)
    u=np.full(a.shape,.5)
    np.testing.assert_array_equal(audit.reference_native_action(a,b,"mode",u),[[0.,1.,.5,0.,1.,.5]])
    np.testing.assert_array_equal(audit.reference_native_action(a,b,"mean",u)[:,:3],[[.25,.75,.5]])
    sampled=audit.reference_native_action(a,b,"sample",u)
    assert np.isfinite(sampled).all() and np.all((sampled>0)&(sampled<1))

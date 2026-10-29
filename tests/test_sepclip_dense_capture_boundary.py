"""Dense snapshot preserves autoreset/final-observation and GAE time-limit semantics."""
import numpy as np
import torch
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as trainer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


def test_dense_time_limit_capture_and_trace_boundary():
    configure_runtime()
    assert torch.cuda.is_available()
    envs=trainer.make_mujoco_vector_env('HalfCheetah-v4',2,num_threads=2)
    try:
        raw,_=envs.reset(seed=1)
        norm=VectorObsNorm(2,(17,)); rewards=VectorRewardNorm(2,.99)
        norm.normalize(raw)
        envs._episode_lengths[0]=999
        envs._limits[0]._elapsed_steps=999
        capture=trainer.TransitionCapture(envs,2)
        masks=[]; normalized_rewards=[]
        for step in range(2):
            action=np.full((2,6),.3,np.float32)
            capture.before(step,envs,action,norm)
            raw,reward,term,trunc,infos=envs.step(action)
            nr=rewards.normalize(reward,term)
            obs,transition=norm.normalize_step(raw,term,trunc,infos)
            capture.after(step,raw,reward,obs,transition,rewards)
            normalized_rewards.append(nr.copy()); masks.append((term.copy(),trunc.copy()))
        assert masks[0][1].tolist()==[True,False]
        assert masks[1][1].tolist()==[False,False]
        assert capture.arrays['episode_age'][:,0].tolist()==[999,0]
        assert not np.array_equal(capture.arrays['normalized_next_obs'][0,0],
                                  capture.arrays['normalized_transition_obs'][0,0])
        for step in range(2):
            np.testing.assert_array_equal(normalized_rewards[step],
                np.clip(capture.arrays['raw_rewards'][step]/capture.arrays['reward_std'][step],-10,10).astype(np.float32))
        r=torch.tensor(np.stack(normalized_rewards),device='cuda')
        terms=torch.tensor(np.stack([m[0] for m in masks]),device='cuda',dtype=torch.float32)
        truncs=torch.tensor(np.stack([m[1] for m in masks]),device='cuda',dtype=torch.float32)
        values=torch.tensor([[.3,.7],[.4,.8]],device='cuda')
        boots=torch.tensor([[2.,0.],[0.,0.]],device='cuda')
        tail=torch.tensor([.9,1.1],device='cuda')
        fn=get_gae_fn(compiled=True,mode='reduce-overhead')
        adv,ret=fn(r,values,terms,truncs,boots,tail,.99,.95)
        expected=torch.empty_like(adv)
        expected[1]=r[1]+.99*tail-values[1]
        expected[0,0]=r[0,0]+.99*boots[0,0]-values[0,0]
        expected[0,1]=r[0,1]+.99*values[1,1]-values[0,1]+.99*.95*expected[1,1]
        torch.testing.assert_close(adv,expected)
        torch.testing.assert_close(ret,expected+values)
    finally:
        envs.close()

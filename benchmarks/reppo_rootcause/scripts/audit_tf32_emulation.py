"""Emulate TF32 matmuls (inputs rounded to a 10-bit mantissa, FP32 accumulate) in the critic's Linear layers and
measure the error it adds to Q, to within-state contrasts and to dQ/da. v36..v59 defaulted to matmul_precision=high."""
import importlib.util
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
REPO = "/home/marvin/Documents/repositories/cleanrl"
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("v60", f"{REPO}/cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py")
v60 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v60)


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mantissa, exponent = torch.frexp(x)
        return torch.ldexp(torch.round(mantissa * 2048.0) / 2048.0, exponent)  # 1 implicit + 10 explicit bits

    @staticmethod
    def backward(ctx, grad):
        return grad


class TF32Linear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return F.linear(Round.apply(x), Round.apply(self.linear.weight), self.linear.bias)


def convert(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, TF32Linear(child))
        else:
            convert(child)


for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**ck["args"])
    obs_dim, act_dim = ck["obs_means"].shape[1], len(ck["temperature"])
    exact = v60.Critic(obs_dim, act_dim, args, "cpu")
    exact.load_state_dict(ck["critic"])
    tf32 = v60.Critic(obs_dim, act_dim, args, "cpu")
    tf32.load_state_dict(ck["critic"])
    convert(tf32)
    actor = v60.Actor(obs_dim, act_dim, args)
    actor.load_state_dict(ck["actor_state"])
    g = torch.Generator().manual_seed(0)
    obs = 0.6 * torch.randn(4096, obs_dim, generator=g)  # near the data's bulk
    with torch.no_grad():
        mean, std = actor(obs)
        acts = torch.tanh(mean[None] + std[None] * torch.randn(8, 4096, act_dim, generator=g))
        qe, qt = exact.value(obs.expand(8, -1, -1), acts), tf32.value(obs.expand(8, -1, -1), acts)
    ce, ct = qe - qe.mean(0), qt - qt.mean(0)
    a = acts[0].clone().requires_grad_(True)
    (ge,) = torch.autograd.grad(exact.value(obs, a).sum(), a)
    a = acts[0].clone().requires_grad_(True)
    (gt,) = torch.autograd.grad(tf32.value(obs, a).sum(), a)
    print(f"{path.split('/')[-1]}: Q mean {qe.mean():.1f}; TF32 Q error rms {(qt - qe).square().mean().sqrt():.4f}; contrast sd {ce.std():.3f}, "
          f"TF32 contrast error rms {(ct - ce).square().mean().sqrt():.4f}; dQ/da relative error median "
          f"{((gt - ge).norm(dim=-1) / ge.norm(dim=-1)).median():.4f}")

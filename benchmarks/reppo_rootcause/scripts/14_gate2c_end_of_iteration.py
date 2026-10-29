"""Gate 2, addendum: the actor gradient at the END of an iteration (checkpoint actor against the checkpoint's old_actor).

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_gate2c_end_of_iteration.py

14_gate2_spikes.py scores the first actor step of an iteration (old = current policy, nobody outside the KL gate) and
gets pre-clip norms of 0.35 / 1.06 / 1.17; the run logged iteration means of 0.72 / 3.08 / 2.59. The checkpoint holds the
policy after its 32 steps and the anchor it started from, so the last step's loss can be rebuilt on the sampled states:
states inside the gate get -Q + T log pi, states outside get kl * multiplier. Which part carries the norm?
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("common14", Path(__file__).resolve().parent / "14_common.py")
c14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c14)
torch.set_num_threads(1)

for step in c14.STEPS:
    ck, s = c14.load_checkpoint(step), c14.load_states(step)
    actor, critic, trainer, args = ck.actor, ck.critic, ck.trainer, ck.args
    old = torch.nn.Sequential(*[type(m)(*([m.in_features, m.out_features] if isinstance(m, torch.nn.Linear) else [m.normalized_shape[0]] if isinstance(m, torch.nn.RMSNorm) else []))
                                for m in actor.net])
    old.load_state_dict(ck.checkpoint["old_actor"])
    for index, module in enumerate(actor.net):
        if isinstance(module, torch.nn.RMSNorm):
            old[index].eps = module.eps
    obs_all = torch.from_numpy(c14.normalise(s["raw"], s["lane"], ck))
    noise_all = torch.from_numpy(s["noise"])
    params = list(actor.net.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    temperature, multiplier = actor.temperature().detach(), actor.log_kl_multiplier.exp().detach()
    rng = np.random.default_rng(0)
    rows_out = []
    for _ in range(24):
        rows = torch.from_numpy(rng.integers(0, len(obs_all), args.minibatch_size))
        obs, noise = obs_all[rows], noise_all[rows]
        with torch.no_grad():
            old_mean, old_std = trainer.policy_moments(old, obs)
        mean, std = actor(obs)
        pre = mean + std * noise
        action = torch.tanh(pre)
        logprob = -0.5 * noise.square() - std.log() - trainer.LOG_SQRT_2PI - trainer.tanh_log_det(pre)
        objective = -critic.value(obs, action) + (temperature * logprob).sum(-1)
        kl = trainer.gaussian_kl(old_mean, old_std, mean, std)
        inside = kl < args.kl_bound
        zero = torch.zeros_like(kl)
        flat = lambda loss: torch.cat([g.flatten() for g in torch.autograd.grad(loss, params, retain_graph=True)]).double()
        part_in, part_out = flat(torch.where(inside, objective, zero).mean()), flat(torch.where(inside, zero, kl * multiplier).mean())
        total = part_in + part_out
        rows_out.append([float(total.norm()), float(part_in.norm()), float(part_out.norm()), float(part_in @ part_out / (part_in.norm() * part_out.norm())),
                         float(1 - inside.float().mean()), float(kl.mean()), float(kl.median())])
    r = np.median(np.array(rows_out), 0)
    print(f"checkpoint {step}: last-step actor loss on {args.minibatch_size}-row minibatches of the sampled states: total pre-clip norm {r[0]:.2f} "
          f"(objective part on inside-gate states {r[1]:.2f}, KL pull-back on outside-gate states {r[2]:.2f}, cosine between the two {r[3]:+.2f}); "
          f"fraction outside the gate {r[4]:.2f}; KL(old || new) mean {r[5]:.4f}, median {r[6]:.4f}; multiplier {float(multiplier):.2f}; clip {args.max_grad_norm}")
    for parameter in params:
        parameter.requires_grad_(False)

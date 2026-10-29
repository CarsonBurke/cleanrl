# 13: does the untrained tanh-Gaussian head tip HalfCheetah over where Beta / a quieter head does not?
# The "one-second check" of 09 section 1.5, never run. CPU, no learning: fixed (untrained) action
# distributions only. Inverted = cos(root pitch) < 0 (obs[1] is unwrapped). NOT YET RUN (see 13_redteam_plan.md).
import json
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
EPISODES, STEPS = 16, 1000


def v60_actor_net(seed=1):
    """v60's actor constructor order: the first module built after torch.manual_seed(seed)."""
    torch.manual_seed(seed)
    layers = []
    for i, o in ((17, 512), (512, 512)):
        layers += [nn.Linear(i, o), nn.RMSNorm(o, eps=1e-5), nn.SiLU()]
    return nn.Sequential(*layers, nn.Linear(512, 12)).requires_grad_(False)


class RunningNorm:
    def __init__(self):
        self.n, self.mean, self.var = 1e-4, np.zeros(17), np.ones(17)

    def __call__(self, x):
        d = x - self.mean
        tot = self.n + 1
        self.mean = self.mean + d / tot
        self.var = (self.var * self.n + d * (x - self.mean)) / tot
        self.n = tot
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)


def iid(kind, scale):
    def act(rng, obs, state):
        if kind == "tanh":
            return np.tanh(scale * rng.standard_normal(6))
        if kind == "beta":
            return 2.0 * rng.beta(scale, scale, 6) - 1.0
        return rng.uniform(-1, 1, 6)
    return act


def net_policy(log_std_shift=0.0, weight_gain=1.0):
    net = v60_actor_net()
    net[-1].weight.mul_(weight_gain)
    net[-1].bias.mul_(weight_gain)

    def act(rng, obs, state):
        norm = state.setdefault("norm", RunningNorm())
        out = net(torch.from_numpy(norm(obs))[None])[0].numpy()
        mean, log_std = out[:6], out[6:] + log_std_shift
        state.setdefault("std", []).append(np.exp(log_std).mean())
        return np.tanh(mean + np.exp(log_std) * rng.standard_normal(6))
    return act


CONDITIONS = {
    "iid tanh N(0,1.08)": iid("tanh", 1.08),
    "iid tanh N(0,0.80)": iid("tanh", 0.80),
    "iid tanh N(0,0.55)": iid("tanh", 0.55),
    "iid tanh N(0,0.40)": iid("tanh", 0.40),
    "iid Beta(1.69,1.69)": iid("beta", 1.69),
    "v60 init net (seed 1)": net_policy(),
    "v60 init net, log-std bias -0.6": net_policy(log_std_shift=float(np.log(0.55))),
    "v60 init net, last layer x0.01, sigma 0.55": net_policy(float(np.log(0.55)), 0.01),
    "v60 init net, last layer x0.01, sigma 1.0": net_policy(0.0, 0.01),
}


def main():
    env = gym.make("HalfCheetah-v4")
    rows = {}
    for name, act in CONDITIONS.items():
        rng = np.random.default_rng(13)
        inverted_steps, ever, end, first, returns, sat, state = 0, 0, 0, [], [], [], {}
        for episode in range(EPISODES):
            obs, _ = env.reset(seed=1000 + episode)
            total, first_step, inv = 0.0, None, False
            for step in range(STEPS):
                action = act(rng, obs, state)
                sat.append(float((np.abs(action) > 0.9).mean()))
                obs, reward, term, trunc, _ = env.step(action.astype(np.float32))
                total += reward
                inv = np.cos(obs[1]) < 0.0  # pitch is unwrapped: a full somersault is upright again
                inverted_steps += inv
                if inv and first_step is None:
                    first_step = step
            ever += first_step is not None
            end += inv
            if first_step is not None:
                first.append(first_step)
            returns.append(total)
        rows[name] = dict(inverted_frac=inverted_steps / (EPISODES * STEPS), episodes_ever_inverted=ever,
                          episodes_end_inverted=end, median_first_inversion=float(np.median(first)) if first else None,
                          mean_return=float(np.mean(returns)), frac_abs_a_gt_0p9=float(np.mean(sat)),
                          mean_std=float(np.mean(state["std"])) if "std" in state else None)
        print(name, json.dumps(rows[name]), flush=True)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as handle:
            json.dump(dict(episodes=EPISODES, steps=STEPS, conditions=rows), handle, indent=1)


if __name__ == "__main__":
    main()

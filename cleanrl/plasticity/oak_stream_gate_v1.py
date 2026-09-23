"""The Oak lab "learning from experience" linear stream, with GatePolar beside SGD, Adam and IDBD.

Stream (reference/oak_learning_from_experience.md): 4096 Bernoulli(0.01) inputs; the target is
+1 when the first input is one, else 0; with probability .01 a random +-1 is added; Gaussian noise
of variance 5 on every target. Linear predictor, zero init, one step per sample, predict before
update. Only the first weight is learnable; everything else is noise absorption.

Learners: sgd, adam (torch.optim.Adam), idbd (Sutton 1992, per-weight log step sizes, meta rate
theta), gate_polar (cleanrl.shared.gate_polar.GatePolar on the single Linear; the output layer is
an Adam group, so on a linear model this is the resistance gate on Adam's direction).

Saved per learner: clean-target MSE over the last 20% of the stream, the final weight vector, and
the predictions / targets over a late window, for the blog-style chart (`--window`).
"""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from cleanrl.shared.gate_polar import GatePolar


@dataclass
class Args:
    seed: int = 1
    input_dim: int = 4096
    rate: float = 0.01
    flip_rate: float = 0.01
    noise_variance: float = 5.0
    steps: int = 200000
    window: int = 1500
    sgd_lrs: tuple[float, ...] = (0.003, 0.01, 0.03)
    adam_lrs: tuple[float, ...] = (0.001, 0.003, 0.01)
    idbd_thetas: tuple[float, ...] = (0.001, 0.01)
    idbd_alpha0: float = 0.01
    gate_lrs: tuple[float, ...] = (0.001, 0.003, 0.01)
    gate_decays: tuple[float, ...] = (0.3, 1.0)
    gate_beta: float = 0.99999
    output: str = ""


class Stream:
    def __init__(self, a, device):
        self.a, self.device = a, device
        self.gen = torch.Generator(device=device).manual_seed(a.seed)

    def draw(self):
        a = self.a
        x = (torch.rand(a.input_dim, generator=self.gen, device=self.device) < a.rate).float()
        flip = (torch.rand((), generator=self.gen, device=self.device) < a.flip_rate).float()
        sign = torch.where(torch.rand((), generator=self.gen, device=self.device) < 0.5, -1.0, 1.0)
        clean = x[0]
        noisy = clean + flip * sign + math.sqrt(a.noise_variance) * torch.randn((), generator=self.gen, device=self.device)
        return x, clean, noisy


class Manual:
    """SGD and IDBD on a weight vector; predict(x) then update(x, error)."""

    def __init__(self, kind, a, device, lr=0.0, theta=0.0):
        self.kind, self.theta = kind, theta
        self.w = torch.zeros(a.input_dim, device=device)
        self.lr = lr
        if kind == "idbd":
            self.beta = torch.full((a.input_dim,), math.log(a.idbd_alpha0), device=device)
            self.h = torch.zeros(a.input_dim, device=device)

    def predict(self, x):
        return self.w @ x

    @torch.no_grad()
    def update(self, x, error):
        # error = target - prediction; gradient of half squared error is -error x
        if self.kind == "sgd":
            self.w.add_(self.lr * error * x)
            return
        self.beta.add_(self.theta * error * x * self.h)
        alpha = self.beta.exp()
        self.w.add_(alpha * error * x)
        self.h.mul_((1 - alpha * x * x).clamp(min=0)).add_(alpha * error * x)


class Torch:
    def __init__(self, kind, a, device, lr, decay=0.0):
        self.linear = torch.nn.Linear(a.input_dim, 1, bias=False, device=device)
        torch.nn.init.zeros_(self.linear.weight)
        if kind == "adam":
            self.opt = torch.optim.Adam(self.linear.parameters(), lr=lr)
        else:
            self.opt = GatePolar([{"params": list(self.linear.parameters()), "polar": False}], lr=lr, eps=1e-8,
                                 weight_decay=decay, gate_beta=a.gate_beta)

    @property
    def w(self):
        return self.linear.weight.detach()[0]

    def predict(self, x):
        self.prediction = self.linear(x)[0]
        return self.prediction.detach()

    def update(self, x, error):
        loss = 0.5 * (self.prediction - (self.prediction.detach() + error)).square()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()


def learners(a, device):
    out = {}
    for lr in a.sgd_lrs:
        out[f"sgd_lr{lr:g}"] = Manual("sgd", a, device, lr=lr)
    for theta in a.idbd_thetas:
        out[f"idbd_theta{theta:g}"] = Manual("idbd", a, device, theta=theta)
    for lr in a.adam_lrs:
        out[f"adam_lr{lr:g}"] = Torch("adam", a, device, lr)
    for lr in a.gate_lrs:
        for decay in a.gate_decays:
            out[f"gate_polar_lr{lr:g}_wd{decay:g}"] = Torch("gate_polar", a, device, lr, decay)
    return out


def main():
    a = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    stream = Stream(a, device)
    pool = learners(a, device)
    names = list(pool)
    score_from = int(0.8 * a.steps)
    window_from = a.steps - a.window
    squared = {n: torch.zeros((), device=device) for n in names}
    window = {n: [] for n in names}
    targets, cleans, features = [], [], []
    started = time.perf_counter()
    for t in range(a.steps):
        x, clean, noisy = stream.draw()
        in_window = t >= window_from
        if in_window:
            targets.append(noisy); cleans.append(clean); features.append(x[0])
        for n, learner in pool.items():
            p = learner.predict(x)
            if t >= score_from:
                squared[n] += (p - clean).square()
            if in_window:
                window[n].append(p)
            learner.update(x, noisy - p)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {"args": asdict(a), "seconds": seconds, "learners": {}}
    for n, learner in pool.items():
        w = learner.w
        result["learners"][n] = {
            "clean_mse_last20": (squared[n] / (a.steps - score_from)).item(),
            "w0": w[0].item(), "distractor_abs_mean": w[1:].abs().mean().item(), "distractor_abs_max": w[1:].abs().max().item(),
            "window_predictions": torch.stack(window[n]).tolist()}
    result["window"] = {"targets": torch.stack(targets).tolist(), "clean": torch.stack(cleans).tolist(),
                        "feature0": torch.stack(features).tolist(), "start": window_from}
    root = Path(a.output or f"runs/OakStreamGate__v1__{a.seed}__{time.time_ns()}")
    root.mkdir(parents=True)
    (root / "results.json").write_text(json.dumps(result) + "\n")
    print(f"{root} ({seconds:.0f}s)")
    for n, r in result["learners"].items():
        print(f"  {n:28s} clean mse {r['clean_mse_last20']:.5f}  w0 {r['w0']:+.3f}  |w_distractor| mean {r['distractor_abs_mean']:.4f} max {r['distractor_abs_max']:.3f}")


if __name__ == "__main__":
    main()

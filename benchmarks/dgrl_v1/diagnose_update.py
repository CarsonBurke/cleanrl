"""Compare actor location updates on one frozen TD7 latent-Q checkpoint.

Every rule sees the same actor, critic, encoder and replay states, so differences are the
update rule's alone. Location only: log_std is detached in every surrogate.

  dpg         pathwise J^T dQ/da at the mode (the critic's own local improvement direction)
  pg4, pg32   score-function location update, modal reference, antithetic draws
  delight4    v10's trained location update (mean scope, raw gate)
  dbu*_tauT   DGRL v1's target regression, M draws plus the mode, temperature T

Reported per rule:
  act_cos     median per-state cosine between the action-space step and dQ/da at the mode
  dpg_cos     cosine of the 256-state actor-parameter gradient with dpg on the same states
  consistency mean pairwise cosine between gradients from disjoint 256-state batches
  gain/kl     held-out mean Q gain of a step along the 256-state gradient, scaled to a
              mean-term KL of 1e-3, averaged over batches (normalised by dpg's gain)
  adam_gain   same, but the step is sign(g) (Adam's first step), scaled to the same KL
"""
import argparse
import json

import torch

from cleanrl.td7.delight import td7_latentq_delight_reference_v10 as v10
from cleanrl.td7.dgrl import td7_latentq_dgrl_v1 as dgrl


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    models = checkpoint["agent"]["models"]
    action_dim = models["actor"]["l3.weight"].shape[0]
    state_dim = models["actor"]["l0.weight"].shape[1]
    zs_dim = args["zs_dim"] if args["actor_sale"] else 0
    actor = v10.Actor(state_dim, action_dim, zs_dim, args["hidden_dim"]).to(device)
    critic = v10.PreActionLatentTwinCritic(state_dim, action_dim, args["zs_dim"], args["hidden_dim"]).to(device)
    encoder = v10.Encoder(state_dim, action_dim, args["zs_dim"], args["hidden_dim"]).to(device)
    actor.load_state_dict(models["actor"])
    critic.load_state_dict(models["critic"])
    encoder.load_state_dict(models["fixed_encoder"])
    for module in (critic, encoder):
        module.requires_grad_(False)
    states = checkpoint["agent"]["storage"][:, :state_dim].to(device)
    return actor, critic, encoder, states, checkpoint["global_step"]


class Probe:
    def __init__(self, actor, critic, encoder):
        self.actor, self.critic, self.encoder = actor, critic, encoder
        self.parameters = list(actor.parameters())

    def point_q(self, state, zs, actions):
        """Twin-mean point head at (count, batch, action) actions; differentiable in actions."""
        count, batch, action_dim = actions.shape
        flat = actions.reshape(-1, action_dim)
        states = state.expand(count, -1, -1).reshape(-1, state.shape[-1])
        z = zs.expand(count, -1, -1).reshape(-1, zs.shape[-1])
        latent = torch.cat([flat, torch.zeros_like(flat)], 1)
        return self.critic(states, latent, flat, self.encoder.zsa(z, flat), z).mean(-1).reshape(count, batch)

    def flat_grad(self, loss):
        return torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            loss, self.parameters, allow_unused=True, materialize_grads=True)])

    def rule(self, name, state, zs):
        """(actor-parameter gradient of the descent loss, per-state action-space ascent step)."""
        mu, log_std = self.actor(state, zs)
        sigma = log_std.detach().exp()
        if name == "dpg":
            a = mu.detach().requires_grad_()
            dq = torch.autograd.grad(self.point_q(state, zs, a[None]).sum(), a)[0]
            return self.flat_grad(-(mu * dq).sum(-1).mean()), dq
        kind, _, rest = name.partition("_tau")
        samples = int("".join(ch for ch in kind if ch.isdigit()))
        eps = torch.randn(samples // 2, *mu.shape, device=mu.device)
        eps = torch.cat([eps, -eps])
        with torch.no_grad():
            draws = mu.detach() + sigma * eps
            if kind.startswith("dbu"):  # dbuN: L2 (v1); dbumN: Mahalanobis, the weighted-MLE M-step
                candidates = torch.cat([draws, mu.detach()[None]])
                q = self.point_q(state, zs, candidates.clamp(-1, 1))
                target, _, _ = dgrl.dbu_targets(candidates, q, samples, float(rest), True)
                step = target - mu.detach()
                if kind.startswith("dbum"):
                    step = step / sigma.square()
            else:
                q = self.point_q(state, zs, torch.cat([draws, mu.detach()[None]]).clamp(-1, 1))
                advantage = q[:-1] - q[-1]
                weight = torch.ones_like(advantage)
                if kind.startswith("delight"):
                    weight, _, _ = v10.delight_weight(advantage, eps, objective="delight", scale="raw")
                step = ((weight * advantage)[..., None] * eps * sigma).mean(0)
        if kind.startswith("dbum"):
            loss = 0.5 * ((mu - target) / sigma).square().sum(-1).mean()
        elif kind.startswith("dbu"):
            loss = 0.5 * (mu - target).square().sum(-1).mean()
        else:
            log_prob = v10.gaussian_log_probability(draws, mu, log_std.detach())
            loss = -((weight * advantage) * log_prob).mean()
        return self.flat_grad(loss), step

    @torch.no_grad()
    def gain(self, direction, held_state, held_zs, kl=1e-3):
        """Held-out mean Q gain of theta - eta*direction with mean-term KL = kl (bisection on eta)."""
        mu0, log_std0 = self.actor(held_state, held_zs)
        q0 = self.point_q(held_state, held_zs, mu0[None]).mean()
        saved = [p.detach().clone() for p in self.parameters]
        offsets = torch.split(direction, [p.numel() for p in self.parameters])

        def apply(eta):
            for p, s, d in zip(self.parameters, saved, offsets):
                p.copy_(s - eta * d.view_as(p))
            mu, _ = self.actor(held_state, held_zs)
            return mu, (0.5 * ((mu - mu0) / log_std0.exp()).square().sum(-1)).mean()

        low, high = 0.0, 1.0
        while apply(high)[1] < kl and high < 1e6:
            high *= 2
        for _ in range(40):
            mid = 0.5 * (low + high)
            low, high = (mid, high) if apply(mid)[1] < kl else (low, mid)
        mu, _ = apply(low)
        q1 = self.point_q(held_state, held_zs, mu[None]).mean()
        for p, s in zip(self.parameters, saved):
            p.copy_(s)
        return (q1 - q0).item()


def cosine(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="benchmarks/dgrl_v1/diagnose_update_v2.json")
    options = parser.parse_args()
    torch.manual_seed(1)
    device = torch.device("cuda")
    rules = ["dpg", "pg4", "delight4", "pg32", "delight32", "pg128", "dbu32_tau1.0",
             "dbum32_tau0.3", "dbum32_tau1.0", "dbum32_tau3.0", "dbum32_tau10.0", "dbum128_tau1.0"]
    report = {}
    for path in options.checkpoints:
        actor, critic, encoder, states, step = load(path, device)
        probe = Probe(actor, critic, encoder)
        held = states[torch.randint(len(states), (8192,), device=device)]
        held_zs = encoder.zs(held)
        index = torch.randint(len(states), (options.batches, options.batch_size), device=device)
        results = {}
        for name in rules:
            torch.manual_seed(2)
            grads, act_cos, dpg_cos, gains, adam_gains = [], [], [], [], []
            for b in range(options.batches):
                state = states[index[b]]
                zs = encoder.zs(state)
                torch.manual_seed(100 + b)
                g, a_step = probe.rule(name, state, zs)
                g_dpg, dq = probe.rule("dpg", state, zs)
                grads.append(g)
                act_cos.append(cosine(a_step, dq).median().item())
                dpg_cos.append(cosine(g, g_dpg).item())
                if b < 8:
                    gains.append(probe.gain(g, held, held_zs))
                    adam_gains.append(probe.gain(g.sign(), held, held_zs))
            stacked = torch.stack(grads)
            unit = stacked / stacked.norm(dim=-1, keepdim=True)
            pairwise = unit @ unit.T
            n = len(grads)
            results[name] = dict(
                act_cos=sum(act_cos) / n, dpg_cos=sum(dpg_cos) / n,
                consistency=((pairwise.sum() - n) / (n * (n - 1))).item(),
                gain=sum(gains) / len(gains), adam_gain=sum(adam_gains) / len(adam_gains))
        reference = results["dpg"]["gain"]
        for values in results.values():
            values["gain_vs_dpg"] = values["gain"] / reference
            values["adam_gain_vs_dpg"] = values["adam_gain"] / reference
        report[f"{path}@{step}"] = results
        print(f"\n{path} @ {step}")
        print(f"{'rule':16s} {'act_cos':>8s} {'dpg_cos':>8s} {'consist':>8s} {'gain':>9s} {'g/dpg':>7s} {'adam/dpg':>8s}")
        for name, v in results.items():
            print(f"{name:16s} {v['act_cos']:8.3f} {v['dpg_cos']:8.3f} {v['consistency']:8.3f} "
                  f"{v['gain']:9.4f} {v['gain_vs_dpg']:7.3f} {v['adam_gain_vs_dpg']:8.3f}")
    with open(options.out, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()

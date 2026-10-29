"""How much do independent Q estimates agree on the within-state ranking of DBU draws?

On a frozen DGRL Mahalanobis v2 checkpoint, score M antithetic draws plus the mode at replay
states with four point-Q estimators: the two live twins (A, B) and the two hard-updated target
twins (A', B'). For an estimator pair, d^j are within-state centered deviations; the per-state
signal power is S = mean_k d^A d^B (noise cancels if their errors are independent), the noise
power of their average is N = mean_k (d^A - d^B)^2 / 4, and the Kalman gain is S+ / (S+ + N).
"""
import argparse
import json

import torch

from cleanrl.td7.dgrl import td7_latentq_dgrl_mahalanobis_v2 as base


def point_q(critic, encoder, state, zs, actions):
    count, batch, action_dim = actions.shape
    flat = actions.reshape(-1, action_dim)
    states = state.expand(count, -1, -1).reshape(-1, state.shape[-1])
    z = zs.expand(count, -1, -1).reshape(-1, zs.shape[-1])
    latent = torch.cat([flat, torch.zeros_like(flat)], 1)
    return critic(states, latent, flat, encoder.zsa(z, flat), z).reshape(count, batch, 2)


def gain(dx, dy):
    signal = (dx * dy).mean(0).clamp(min=0)
    noise = (dx - dy).square().mean(0) / 4
    return signal / (signal + noise).clamp(min=1e-12), noise / (dx + dy).square().mean(0).div(4).clamp(min=1e-12)


def summary(x):
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device))
    return {"mean": x.mean().item(), "p10": q[0].item(), "p50": q[1].item(), "p90": q[2].item()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--states", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = checkpoint["args"]
    models = checkpoint["agent"]["models"]
    action_dim, state_dim = models["actor"]["l3.weight"].shape[0], models["actor"]["l0.weight"].shape[1]
    actor = base.Actor(state_dim, action_dim, a["zs_dim"] if a["actor_sale"] else 0, a["hidden_dim"]).to(device)
    actor.load_state_dict(models["actor"])
    critics = {}
    for name in ("critic", "critic_target"):
        critics[name] = base.PreActionLatentTwinCritic(state_dim, action_dim, a["zs_dim"], a["hidden_dim"]).to(device)
        critics[name].load_state_dict(models[name])
    encoder = base.Encoder(state_dim, action_dim, a["zs_dim"], a["hidden_dim"]).to(device)
    encoder.load_state_dict(models["fixed_encoder"])
    storage = checkpoint["agent"]["storage"]
    torch.manual_seed(0)
    report = {"global_step": checkpoint["global_step"]}
    # Uniform replay states and the newest rollout's states (where the policy currently acts).
    for split, rows in (("replay", storage[torch.randint(len(storage), (args.states,))]),
                        ("recent", storage[-args.states:])):
        state = rows[:, :state_dim].to(device)
        zs = encoder.zs(state)
        mu, log_std = actor(state, zs)
        eps = torch.randn(args.samples // 2, len(state), action_dim, device=device)
        candidates = torch.cat([mu + log_std.exp() * torch.cat([eps, -eps]), mu[None]])
        q = {}
        for name, critic in critics.items():
            out = point_q(critic, encoder, state, zs, candidates.clamp(-1, 1))
            q[name + "_A"], q[name + "_B"] = out[..., 0], out[..., 1]
        d = {k: v - v.mean(0) for k, v in q.items()}
        rms = ((d["critic_A"] + d["critic_B"]) / 2).square().mean().sqrt()
        result = {"rms_mean_deviation": rms.item(), "sigma_geomean": log_std.mean().exp().item()}
        pairs = {"twin": (d["critic_A"], d["critic_B"]),
                 "target_twin": (d["critic_target_A"], d["critic_target_B"]),
                 "live_vs_target": ((d["critic_A"] + d["critic_B"]) / 2, (d["critic_target_A"] + d["critic_target_B"]) / 2)}
        for name, (x, y) in pairs.items():
            k, noise_fraction = gain(x, y)
            corr = ((x * y).mean(0) / (x.square().mean(0) * y.square().mean(0)).sqrt().clamp(min=1e-12))
            result[name] = {"kalman_gain": summary(k), "noise_fraction": summary(noise_fraction),
                            "rank_corr_proxy": summary(corr)}
        # Does the twin gain track how much the action matters at a state (signal scale)?
        k_twin, _ = gain(*pairs["twin"])
        spread = ((d["critic_A"] + d["critic_B"]) / 2).square().mean(0).sqrt()
        order = spread.argsort()
        quintiles = [k_twin[idx].mean().item() for idx in order.chunk(5)]
        result["twin_gain_by_spread_quintile"] = quintiles
        report[split] = result
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()

"""Independent numerical contracts and real-history NorMuon counterfactuals.

Run through mlq. This is a diagnostic, not a shortened training experiment.
Axis changes reset second moments for *all* equalizer variants; incompatible
row statistics cannot be converted to column statistics from a checkpoint.
"""
import argparse
import copy
import functools
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import awr_update_bench as bench
import diagnose_sepclip_normuon as probe
from diagnose_sepclip_normuon_history import restore_muon
from cleanrl import ppo_continuous_action_sepclip_normuon_split_diagnostic_v2 as m


def update(p, buf, second, grad, lr, momentum, beta2, scale, *, axis, norm):
    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
    direction = m.polar_express((1 - momentum) * grad + momentum * buf).float()
    if norm == "plain":
        p.sub_(lr * scale * direction)
        return
    second.mul_(beta2).add_(direction.square().mean(axis, keepdim=True), alpha=1 - beta2)
    adapted = direction / second.clamp_min(1e-10).sqrt()
    target = (0.2 * math.sqrt(p.shape[-2] * p.shape[-1]) if norm == "paper"
              else scale * direction.norm(dim=(-2, -1), keepdim=True))
    p.sub_(lr * target * adapted / adapted.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10))


@torch.no_grad()
def numerical_contracts():
    """Independent analytic spectral oracle; actual SiTU shapes and state roundtrip."""
    results = []
    for rows, cols in ((43, 17), (43, 64), (64, 43)):
        rank = min(rows, cols)
        left, _ = torch.linalg.qr(torch.randn(rows, rank, device="cuda", dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(cols, rank, device="cuda", dtype=torch.float64))
        singular = torch.linspace(0.15, 1, rank, device="cuda", dtype=torch.float64)
        gradient = ((left * singular) @ right.T).float()
        # Apply the scalar polynomial to singular values, without calling the
        # production matrix polynomial to construct the expected polar factor.
        spectrum = singular / (singular.norm() * 1.02 + 1e-6)
        for a, b, c in m.POLAR_EXPRESS:
            spectrum = a * spectrum + b * spectrum**3 + c * spectrum**5
        expected = ((left * spectrum) @ right.T).float()
        actual = m.polar_express(gradient).float()
        relative = float((actual - expected).norm() / expected.norm())
        assert relative < 0.12, (rows, cols, relative)
        assert torch.count_nonzero(m.polar_express(torch.zeros_like(gradient))) == 0
        # Check persistent state and gradient ownership against independent
        # arithmetic across multiple nonparallel gradients and scheduled LRs.
        p = torch.nn.Parameter(torch.randn(rows, cols, device="cuda"))
        opt = m.PolarExpressNorMuon([p])
        momentum = torch.zeros(1, rows, cols, device="cuda")
        second = torch.zeros(1, rows, 1, device="cuda")
        errors = []
        for lr in (0.005, 0.0025, 0.001):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            before = p.detach().clone()
            momentum = 0.95 * momentum + 0.05 * grad[None]
            v = m.polar_express(0.05 * grad[None] + 0.95 * momentum).float()
            second = 0.95 * second + 0.05 * v.square().mean(-1, keepdim=True)
            adapted = v / second.clamp_min(1e-10).sqrt()
            displacement = -lr * math.sqrt(max(1, rows / cols)) * adapted * (
                v.norm() / adapted.norm().clamp_min(1e-10))
            opt.set_lr(lr)
            opt.step()
            error = float(((p - before) - displacement[0]).norm() / displacement.norm())
            errors.append(error)
            assert error < 0.12, error
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
        torch.testing.assert_close(opt.state[p]["momentum_buffer"], momentum)
        # Reproduce the stock optimizer's stale references after load_state_dict.
        restored = m.PolarExpressNorMuon([torch.nn.Parameter(p.detach().clone())])
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        stale = restored.stacks[0][2] is not restored.state[restored.param_groups[0]["params"][0]]
        assert stale
        restore_muon(restored, copy.deepcopy(opt.state_dict()))
        assert restored.stacks[0][2] is restored.state[restored.param_groups[0]["params"][0]]
        results.append(dict(shape=[rows, cols], analytic_polar_relative_error=relative,
                            multistep_relative_errors=errors, stale_state_after_standard_load=stale))
    return results


def replay(path, writer):
    snapshot = torch.load(path, map_location="cuda", weights_only=False)
    args = SimpleNamespace(**snapshot["args"])
    assert args.num_minibatches == 1 and args.normuon_network == "actor"
    assert args.target_kl is None, "replay requires the trainer's full epoch count"
    assert args.clip_coef == args.clip_coef_upper == 0.2, "probe evaluator assumes symmetric 0.2 clipping"
    envs = m.make_training_env(args, "normuon_audit")
    try:
        agent = m.Agent(envs).cuda()
    finally:
        envs.close()
    agent.load_state_dict(snapshot["model"])
    base = copy.deepcopy(agent.state_dict())
    train = snapshot["batch"]
    with torch.no_grad():
        train["alpha"], train["beta"], _ = agent.get_policy_and_value(train["obs"])
        train["advantages_mc"] = train["advantages"]
        chunks, returns = bench.collect(m, agent, snapshot, 3, "cuda", 1)
        for chunk in chunks:
            chunk["values"] = agent.get_value(chunk["obs"]).flatten()
            chunk["returns"] = chunk["values"] + chunk["advantages"]
    hold = bench.concatenate(chunks)
    loss = torch.compile(lambda obs, native, old, adv, ret, values:
                         m.ppo_loss(agent, obs, native, old, adv, ret, values, args),
                         fullgraph=True, options={"triton.cudagraphs": False})
    names = {id(p): name for name, p in agent.named_parameters()}
    cases = []
    # Reset controls isolate the new axis from the change of variance history.
    # Matrix/head freezes isolate their interaction without changing targets.
    for kind in ("actual", "rows_reset", "long_axis_reset", "projection_axis_reset",
                 "paper_rms_reset", "plain_muon", "heads_only", "matrices_only"):
        agent.load_state_dict(base)
        muon, adam = m.make_optimizers(agent, args)
        restore_muon(muon, copy.deepcopy(snapshot["matrix_optimizer"]))
        adam.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        if kind.endswith("reset"):
            for _, ps, state, _ in muon.stacks:
                rows, cols = ps[0].shape
                axis = (-2 if kind == "long_axis_reset" and rows < cols else -1)
                if kind == "projection_axis_reset":
                    axis = -2 if ".down." in names[id(ps[0])] else -1
                assert all((".down." in names[id(p)]) == (".down." in names[id(ps[0])]) for p in ps)
                state["second"] = torch.zeros((len(ps), rows, 1) if axis == -1
                                               else (len(ps), 1, cols), device="cuda")
            # Compiled dispatch is by bank shape/axis, preserving stacked updates.
            funcs = {}
            def dispatch(p, buf, second, grad, lr, momentum, beta2, scale):
                axis = -1 if second.shape[-1] == 1 else -2
                key = (axis, p.shape)
                if key not in funcs:
                    funcs[key] = torch.compile(functools.partial(update, axis=axis,
                        norm="paper" if kind == "paper_rms_reset" else "polar"), fullgraph=True)
                funcs[key](p, buf, second, grad, lr, momentum, beta2, scale)
            muon.update = dispatch
        elif kind == "plain_muon":
            muon.update = torch.compile(functools.partial(update, axis=-1, norm="plain"), fullgraph=True)
        epochs = []
        for epoch in range(args.update_epochs):
            probe.gradients(agent, train, loss)
            torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()), args.max_grad_norm, foreach=True)
            torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()), args.max_grad_norm, foreach=True)
            if kind != "heads_only":
                muon.step()
            if kind == "matrices_only":
                # Critic Adam remains active; freeze only actor auxiliary parameters.
                for p in agent.actor.parameters():
                    if id(p) not in {id(q) for g in muon.param_groups for q in g["params"]}:
                        p.grad = None
            adam.step()
            agent.normalize_matrices()
            train_metrics, hold_metrics = probe.evaluate(agent, train), probe.evaluate(agent, hold)
            epochs.append(dict(epoch=epoch + 1, train=train_metrics, hold=hold_metrics))
            for tag in ("exact_kl", "surrogate_gain", "mc_gain"):
                writer.add_scalar(f"{snapshot['global_step']}/{kind}/{tag}", hold_metrics[tag], epoch + 1)
        cases.append(dict(kind=kind, epochs=epochs,
                          heldout_chunks=[probe.evaluate(agent, chunk) for chunk in chunks]))
        print(json.dumps(dict(step=snapshot["global_step"], kind=kind, final=epochs[-1])), flush=True)
    return dict(snapshot=str(path), step=snapshot["global_step"], cases=cases, rollout_returns=returns,
                limits=["Local counterfactuals, not fresh-training returns.",
                        "Changed axes use reset statistics; compare with rows_reset.",
                        "Sequential held-out chunks are not independent training seeds.",
                        "No matched-KL inference from a single learning rate.",
                        "Train MC duplicates GAE because actual MC targets were not saved."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    m.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(1)
    result = dict(numerical=numerical_contracts(), snapshots=[])
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(result, indent=2) + "\n")
    with SummaryWriter(f"runs/HalfCheetah-v4__sepclip_normuon_reference_audit__1__{int(time.time())}") as writer:
        for path in cli.snapshots:
            result["snapshots"].append(replay(path, writer))
            cli.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

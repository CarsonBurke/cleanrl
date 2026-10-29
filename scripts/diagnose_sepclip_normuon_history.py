"""Replay actual PPO batches and NorMuon history; queue all GPU work via mlq."""
import argparse
import copy
import functools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import awr_update_bench as bench
import diagnose_sepclip_normuon as probe
from cleanrl import ppo_continuous_action_sepclip_normuon_split_diagnostic_v2 as m


def restore_muon(opt, saved):
    opt.load_state_dict(saved)
    # Optimizer.load_state_dict replaces its dictionaries; rebuild references
    # held by the fixed-stack updater so replay uses the saved state and LR.
    opt.stacks.clear()
    for group in opt.param_groups:
        by_shape = {}
        for p in group["params"]:
            by_shape.setdefault(p.shape, []).append(p)
        for shape, ps in by_shape.items():
            opt.stacks.append((group, ps, opt.state[ps[0]], max(1., shape[0] / shape[1]) ** .5))


def replay(path, writer):
    snapshot = torch.load(path, map_location="cuda", weights_only=False)
    args = SimpleNamespace(**snapshot["args"])
    envs = m.make_training_env(args, "history_probe")
    try:
        agent = m.Agent(envs).cuda()
    finally:
        envs.close()
    agent.load_state_dict(snapshot["model"])
    base = copy.deepcopy(agent.state_dict())
    train = snapshot["batch"]
    with torch.no_grad():
        train["alpha"], train["beta"], _ = agent.get_policy_and_value(train["obs"])
        # MC targets were not stored in the actual training batch.
        train["advantages_mc"] = train["advantages"]
        chunks, _ = bench.collect(m, agent, snapshot, 3, "cuda", 1)
        for chunk in chunks:
            chunk["values"] = agent.get_value(chunk["obs"]).flatten()
            chunk["returns"] = chunk["values"] + chunk["advantages"]
    hold = bench.concatenate(chunks)
    loss_fn = torch.compile(lambda obs, native, old, adv, returns, values:
                            m.ppo_loss(agent, obs, native, old, adv, returns, values, args),
                            fullgraph=True, options={"triton.cudagraphs": False})
    initial = probe.gradients(agent, train, loss_fn)
    hold_grads = probe.gradients(agent, hold, loss_fn)
    spectral = probe.spectral_measurements(agent, initial, hold_grads)
    results = []
    for kind in ("actual_history", "half_lr", "quarter_lr", "zero_momentum", "no_momentum", "svd_full", "top5",
                 "sqrt_spectrum", "gradient_spectrum"):
        agent.load_state_dict(base)
        muon, adam = m.make_optimizers(agent, args)
        adam.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        restore_muon(muon, copy.deepcopy(snapshot["matrix_optimizer"]))
        if kind in ("half_lr", "quarter_lr"):
            for group in muon.param_groups:
                group["lr"].mul_(.5 if kind == "half_lr" else .25)
        if kind == "zero_momentum":
            for state in muon.state.values():
                state["momentum_buffer"].zero_()
        if kind == "no_momentum":
            for group in muon.param_groups:
                group["momentum"] = 0.
        if kind in ("svd_full", "top5", "sqrt_spectrum", "gradient_spectrum"):
            rank = 5 if kind == "top5" else 0
            power = {"svd_full": 0., "top5": 0., "sqrt_spectrum": .5, "gradient_spectrum": 1.}[kind]
            muon.update = torch.compile(functools.partial(probe.spectral_update, rank=rank, power=power),
                                        fullgraph=True)
        epochs = []
        for epoch in range(args.update_epochs):
            grads = probe.gradients(agent, train, loss_fn)
            torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()), args.max_grad_norm, foreach=True)
            torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()), args.max_grad_norm, foreach=True)
            before = {name: p.detach().clone() for name, p in agent.named_parameters()}
            measures = []
            names = {id(p): name for name, p in agent.named_parameters()}
            for group, ps, state, scale in muon.stacks:
                for i, p in enumerate(ps):
                    name, g = names[id(p)], p.grad
                    moment = state["momentum_buffer"][i]
                    nesterov = g * (1 - group["momentum"] ** 2) + moment * group["momentum"] ** 2
                    cosine = lambda x, y: (x*y).sum() / (x.norm()*y.norm()).clamp_min(1e-20)
                    measures.append((name, torch.stack((g.norm(), moment.norm(), cosine(g, moment),
                                                        cosine(g, nesterov)))))
            muon.step()
            adam.step()
            preprojection = {name: p.detach().clone() for name, p in agent.named_parameters()}
            agent.normalize_matrices()
            geometry = []
            for name, vals in measures:
                p = dict(agent.named_parameters())[name]
                actual = p - before[name]
                ambient = preprojection[name] - before[name]
                g = grads[name]
                h = hold_grads[name]
                additional = torch.stack((actual.norm(), ambient.norm(), actual.norm()/ambient.norm().clamp_min(1e-20),
                                           -(actual*g).sum(), -(actual*h).sum(),
                                           -(actual*g).sum()/(actual.norm()*g.norm()).clamp_min(1e-20)))
                numbers = torch.cat((vals, additional)).detach().cpu().tolist()
                geometry.append(dict(parameter=name, **dict(zip((
                    "grad_norm", "momentum_norm", "grad_momentum_cosine", "grad_nesterov_cosine",
                    "projected_step_norm", "ambient_step_norm", "projection_retention", "train_linear_gain",
                    "hold_initial_linear_gain", "projected_descent_cosine"), numbers))))
            train_eval, hold_eval = probe.evaluate(agent, train), probe.evaluate(agent, hold)
            epochs.append(dict(epoch=epoch+1, train=train_eval, hold=hold_eval, geometry=geometry))
            writer.add_scalar(f"{snapshot['global_step']}/{kind}/heldout_gain", hold_eval["surrogate_gain"], epoch+1)
            writer.add_scalar(f"{snapshot['global_step']}/{kind}/exact_kl", hold_eval["exact_kl"], epoch+1)
        results.append(dict(kind=kind, epochs=epochs))
        results[-1]["heldout_chunks_final"] = [probe.evaluate(agent, chunk) for chunk in chunks]
        print(json.dumps(dict(snapshot=str(path), kind=kind, final=epochs[-1])), flush=True)
    return dict(snapshot=str(path), global_step=snapshot["global_step"], spectral=spectral, results=results,
                caveat="Actual training MC gain is a duplicate of GAE gain; only held-out MC gain is a separate target.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    m.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(1)
    results = []
    with SummaryWriter(f"runs/HalfCheetah-v4__sepclip_normuon_history_probe__1__{int(time.time())}") as writer:
        for snapshot in cli.snapshots:
            results.append(replay(snapshot, writer))
            cli.out.parent.mkdir(parents=True, exist_ok=True)
            cli.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

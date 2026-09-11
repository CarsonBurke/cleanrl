"""Attribute fixed-rollout actor gradients by critic-target regime; run through mlq."""

import argparse
import importlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_critic_response import make_agent, emit, rmse
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.rollout_graph import graph_compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="+", type=Path)
    for path in parser.parse_args().snapshots:
        snap = torch.load(path, map_location="cpu", weights_only=False)
        trainer = importlib.import_module(snap["trainer_module"])
        args = trainer.Args(**snap["args"])
        configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
        agent = make_agent(trainer, args, snap, torch.device("cuda"))
        batch = {k: v.to("cuda") for k, v in snap["batch"].items()}
        obs, actions, adv, returns = (batch[k] for k in ("observations", "native_actions", "advantages", "returns"))
        # Groups use physical target values, not quantiles chosen to force equal occupancy.
        masks = torch.stack((returns < 871.7539, (returns >= 871.7539) & (returns < 1020.60876), returns >= 1020.60876))
        params = tuple(agent.actor.parameters())

        def score_losses(obs, actions, adv, masks):
            alpha, beta, _ = agent.get_policy_and_value(obs)
            logp = agent.action_logprob(alpha, beta, actions)
            # At behavior policy r=1, clipped PPO gradient equals this score-function gradient.
            return (-(logp * adv).unsqueeze(0) * masks).mean(-1)

        losses = torch.compile(score_losses, fullgraph=True, options={"triton.cudagraphs": False})(
            obs, actions, adv, masks
        )
        gradients = []
        for i in range(3):
            grad = torch.autograd.grad(losses[i], params, retain_graph=i < 2)
            gradients.append(torch.cat([g.flatten() for g in grad]).double())
        grad = torch.stack(gradients)
        full = grad.sum(0)
        counts = masks.sum(-1)
        print(f'\nsnapshot={path} step={snap["global_step"]}')
        print("gradients are full-batch population-weighted, preclip actor score gradients; no update or new samples")
        for i, name in enumerate(("low_lt872", "transition_872_1021", "high_ge1021")):
            mask = masks[i]
            emit(
                name,
                fraction=mask.double().mean(),
                target_mean=(returns * mask).sum() / counts[i],
                adv_mean=(adv * mask).sum() / counts[i],
                adv_rms=((adv.square() * mask).sum() / counts[i]).sqrt(),
                adv_square_share=(adv.square() * mask).sum() / adv.square().sum(),
                gradient_norm=grad[i].norm(),
                gradient_projection_on_full=(grad[i] * full).sum() / full.square().sum(),
                gradient_cosine_full=(grad[i] * full).sum() / (grad[i].norm() * full.norm()),
            )
        emit("all", full_gradient_norm=full.norm(), sum_group_norms=grad.norm(dim=1).sum())


if __name__ == "__main__":
    main()

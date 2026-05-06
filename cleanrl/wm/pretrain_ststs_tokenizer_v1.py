# Self-supervised STSTS tokenizer pretraining for MuJoCo observation histories.
#
# Trains the same frozen target encoder consumed by
# wm_sde_stateent_twohot_v6_relusq_hlgauss_v1.py. The tokenizer is anchored by
# reconstructing normalized observations from observation tokens while dynamics
# CLS tokens are shaped by current-state, next-state, reward, continue, and
# next-dynamics-token prediction. The RL agent then loads this backbone and
# keeps it frozen, avoiding the moving-target collapse seen with joint training.
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cleanrl.wm.wm_sde_stateent_twohot_v6_relusq_hlgauss_v1 import (
    CONTEXT_LEN,
    DYN_FLAT_DIM,
    DYN_TOKEN_COUNT,
    EMBED_DIM,
    STSTSCLSBackbone,
    WM_CLS_NAMES,
    make_env,
    relusq_mlp,
    relusq_mlp_prenorm_out,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 4
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    save_dir: str = "checkpoints/tokenizers"

    obs_recon_coef: float = 1.0
    dyn_obs_coef: float = 0.5
    next_obs_coef: float = 1.0
    transition_coef: float = 1.0
    reward_coef: float = 0.25
    continue_coef: float = 0.25
    variance_coef: float = 0.01

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class TokenizerTrainer(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = STSTSCLSBackbone(obs_dim, CONTEXT_LEN, WM_CLS_NAMES)
        self.obs_decoder = nn.Linear(EMBED_DIM, 1)
        self.dyn_obs_decoder = relusq_mlp(DYN_FLAT_DIM, obs_dim, 128)
        self.next_obs_head = relusq_mlp(DYN_FLAT_DIM + action_dim, obs_dim, 128)
        self.transition = relusq_mlp_prenorm_out(DYN_FLAT_DIM + action_dim, DYN_FLAT_DIM, 256)
        self.reward_head = relusq_mlp(DYN_FLAT_DIM + action_dim, 1, 128)
        self.continue_head = relusq_mlp(DYN_FLAT_DIM + action_dim, 1, 128)

    def dyn_bundle_from_cls(self, cls):
        return torch.stack([cls[name] for name in WM_CLS_NAMES], dim=1)

    def forward_loss(self, obs_seq, next_obs_seq, actions, rewards, continues, args):
        cls = self.backbone(obs_seq, return_obs_tokens=True)
        next_cls = self.backbone(next_obs_seq, return_obs_tokens=True)
        dyn_bundle = self.dyn_bundle_from_cls(cls)
        next_dyn_bundle = self.dyn_bundle_from_cls(next_cls)
        dyn_flat = dyn_bundle.flatten(1)
        za = torch.cat([dyn_flat, actions], dim=-1)

        obs_target = obs_seq[:, -1]
        next_obs_target = next_obs_seq[:, -1]
        obs_recon = self.obs_decoder(cls["obs_tokens"]).squeeze(-1)
        next_obs_recon = self.obs_decoder(next_cls["obs_tokens"]).squeeze(-1)
        dyn_obs = self.dyn_obs_decoder(dyn_flat)
        next_obs_pred = self.next_obs_head(za)
        next_dyn_pred = self.transition(za).view(-1, DYN_TOKEN_COUNT, EMBED_DIM)
        reward_pred = self.reward_head(za).squeeze(-1)
        continue_pred = self.continue_head(za).squeeze(-1)

        obs_recon_loss = 0.5 * (
            F.mse_loss(obs_recon, obs_target) + F.mse_loss(next_obs_recon, next_obs_target)
        )
        dyn_obs_loss = F.mse_loss(dyn_obs, obs_target)
        next_obs_loss = F.mse_loss(next_obs_pred, next_obs_target)
        transition_loss = F.mse_loss(next_dyn_pred, next_dyn_bundle.detach())
        reward_loss = F.mse_loss(reward_pred, rewards)
        continue_loss = F.binary_cross_entropy_with_logits(continue_pred, continues)
        variance_loss = self.variance_loss(dyn_flat) + self.variance_loss(cls["obs_tokens"].flatten(1))

        loss = (
            args.obs_recon_coef * obs_recon_loss
            + args.dyn_obs_coef * dyn_obs_loss
            + args.next_obs_coef * next_obs_loss
            + args.transition_coef * transition_loss
            + args.reward_coef * reward_loss
            + args.continue_coef * continue_loss
            + args.variance_coef * variance_loss
        )
        metrics = {
            "losses/loss": loss,
            "losses/obs_recon": obs_recon_loss,
            "losses/dyn_obs": dyn_obs_loss,
            "losses/next_obs": next_obs_loss,
            "losses/transition": transition_loss,
            "losses/reward": reward_loss,
            "losses/continue": continue_loss,
            "losses/variance": variance_loss,
            "charts/dyn_std": dyn_flat.std(dim=0).mean(),
        }
        return loss, metrics

    @staticmethod
    def variance_loss(z):
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()


def sample_actions(envs, num_envs):
    return np.stack([envs.single_action_space.sample() for _ in range(num_envs)])


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, False, run_name, args.gamma) for i in range(args.num_envs)])
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    model = TokenizerTrainer(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_history = torch.zeros(args.num_envs, CONTEXT_LEN, obs_dim, device=device)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    obs_history = torch.cat([obs_history[:, 1:], next_obs.unsqueeze(1)], dim=1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"{run_name}.pt"
    global_step = 0
    last_metrics = {}

    for iteration in range(1, args.num_iterations + 1):
        obs_seqs = torch.zeros(args.num_steps, args.num_envs, CONTEXT_LEN, obs_dim, device=device)
        next_obs_seqs = torch.zeros_like(obs_seqs)
        actions = torch.zeros(args.num_steps, args.num_envs, action_dim, device=device)
        rewards = torch.zeros(args.num_steps, args.num_envs, device=device)
        continues = torch.zeros(args.num_steps, args.num_envs, device=device)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_seqs[step] = obs_history
            action_np = sample_actions(envs, args.num_envs)
            action = torch.tensor(action_np, dtype=torch.float32, device=device)
            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action_np)
            done_np = np.logical_or(terminations, truncations)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            obs_history = torch.cat([obs_history[:, 1:], next_obs.unsqueeze(1)], dim=1)
            next_obs_seqs[step] = obs_history
            actions[step] = action
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            continues[step] = torch.tensor(1.0 - done_np.astype(np.float32), dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        b_obs_seqs = obs_seqs.reshape((-1, CONTEXT_LEN, obs_dim))
        b_next_obs_seqs = next_obs_seqs.reshape((-1, CONTEXT_LEN, obs_dim))
        b_actions = actions.reshape((-1, action_dim))
        b_rewards = rewards.reshape(-1)
        b_continues = continues.reshape(-1)
        b_inds = np.arange(args.batch_size)

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start:start + args.minibatch_size]
                loss, metrics = model.forward_loss(
                    b_obs_seqs[mb_inds],
                    b_next_obs_seqs[mb_inds],
                    b_actions[mb_inds],
                    b_rewards[mb_inds],
                    b_continues[mb_inds],
                    args,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                last_metrics = metrics

        for key, value in last_metrics.items():
            writer.add_scalar(key, value.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        print(
            f"global_step={global_step}, tokenizer_loss={last_metrics['losses/loss'].item():.4f}, "
            f"next_obs={last_metrics['losses/next_obs'].item():.4f}, "
            f"transition={last_metrics['losses/transition'].item():.4f}, "
            f"dyn_std={last_metrics['charts/dyn_std'].item():.4f}"
        )
        torch.save(
            {
                "tokenizer_backbone": model.backbone.state_dict(),
                "args": vars(args),
                "global_step": global_step,
            },
            checkpoint_path,
        )

    torch.save(
        {
            "tokenizer_backbone": model.backbone.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        },
        checkpoint_path,
    )
    print(f"saved tokenizer checkpoint: {checkpoint_path}")
    envs.close()
    writer.close()

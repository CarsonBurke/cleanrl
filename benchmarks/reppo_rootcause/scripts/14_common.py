"""Shared loader for the 14 gate scripts: v60 checkpoints and the on-distribution states sampled from them. CPU only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("common07", HERE / "07_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

REPO, OUT = common.REPO, common.OUT
RUN = REPO / "runs/HalfCheetah-v4__reppo_noisefork_v60_base__1__1790713932"
STEPS = (1015424, 4013696, 7913088)
LANES = (0, 4, 7, 11, 14)  # one stochastic episode under each of these lanes' statistics: 15 episodes over 3 checkpoints
PITCH, PITCH_VELOCITY, X_VELOCITY = 1, 10, 8
OBS_NAMES = ("rootz", "pitch", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot", "vx", "vz", "pitch_vel",
             "bthigh_vel", "bshin_vel", "bfoot_vel", "fthigh_vel", "fshin_vel", "ffoot_vel")


def load_checkpoint(step):
    trainer = common.load_module("cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py")
    checkpoint = torch.load(RUN / f"checkpoint_{step}.pt", map_location="cpu", weights_only=False)
    args = SimpleNamespace(**checkpoint["args"])
    actor = trainer.Actor(17, 6, args)
    actor.load_state_dict(checkpoint["actor_state"])
    critic = trainer.Critic(17, 6, args, torch.device("cpu"))
    critic.load_state_dict(checkpoint["critic"])
    actor.eval(), critic.eval()
    for parameter in list(actor.parameters()) + list(critic.parameters()):
        parameter.requires_grad_(False)
    return SimpleNamespace(checkpoint=checkpoint, args=args, actor=actor, critic=critic, trainer=trainer,
                           means=np.asarray(checkpoint["obs_means"]),
                           sds=np.sqrt(np.asarray(checkpoint["obs_variances"]) + checkpoint["obs_epsilon"]))


def normalise(raw, lanes, ck):
    """v60's network input: raw observation through the statistics of the lane that produced it."""
    return np.clip((raw - ck.means[lanes]) / ck.sds[lanes], -ck.args.obs_clip, ck.args.obs_clip).astype(np.float32)


def load_states(step):
    return dict(np.load(OUT / f"14_states_{step}.npz"))

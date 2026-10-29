"""Is the critic's attenuated action dependence a limit of the FIT (optimizer, architecture, input parametrization)
or of the LABELS (noise)? CPU, gymnasium, a few hundred small updates; not a training run.

Labels with a known within-state action dependence: the one-step lookahead L1(s, a) = r(s, a) + gamma V'_ckpt(f(s, a)),
a deterministic function of (s, a) evaluated through the simulator with the frozen checkpoint actor/critic. The
critic class of v60 is fit to L1 at the rollout's single action per state (as the trainer: CE on HL-Gauss labels,
AdamW 3e-4, clip 0.5, 2048-row minibatches), with and without label noise of the trainer's residual size, and is
scored on held-out states at K fresh policy actions, where L1 is known exactly.
"""
import importlib.util
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

torch.set_num_threads(3)
REPO = "/home/marvin/Documents/repositories/cleanrl"
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("v60", f"{REPO}/cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py")
v60 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v60)

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
args = SimpleNamespace(**ck["args"])
lanes, obs_dim = ck["obs_means"].shape
act_dim = len(ck["temperature"])
gamma, K, STEPS, HOLD = args.gamma, 8, 1024, 2  # HOLD lanes are held out
frozen = v60.Critic(obs_dim, act_dim, args, "cpu")
frozen.load_state_dict(ck["critic"])
frozen.requires_grad_(False)
actor = v60.Actor(obs_dim, act_dim, args)
actor.load_state_dict(ck["actor_state"])
actor.requires_grad_(False)
obs_mean = torch.as_tensor(ck["obs_means"], dtype=torch.float32)
obs_inv = torch.as_tensor(1.0 / np.sqrt(ck["obs_variances"] + ck["obs_epsilon"]), dtype=torch.float32)
gen = torch.Generator().manual_seed(0)


def norm(raw, lane):
    return ((torch.as_tensor(raw, dtype=torch.float32) - obs_mean[lane]) * obs_inv[lane]).clamp(-args.obs_clip, args.obs_clip)


@torch.no_grad()
def soft_value(obs, noise):
    """V'(s) with shared policy noise (samples, 1, dims) across rows: smooth in s."""
    mean, std = actor(obs)
    pre = mean[None] + std[None] * noise
    logprob = -0.5 * noise.square() - std.log()[None] - v60.LOG_SQRT_2PI - v60.tanh_log_det(pre)
    q = frozen.value(obs.expand(noise.shape[0], -1, -1), torch.tanh(pre))
    return (q - (actor.temperature() * logprob).sum(-1)).mean(0)


v_noise = torch.randn(32, 1, act_dim, generator=gen)
envs = [gym.make("HalfCheetah-v4", max_episode_steps=100000) for _ in range(lanes)]  # no time limit: one long branchable stream
raw = np.stack([env.reset(seed=4242 + i)[0] for i, env in enumerate(envs)])
lane_ids = torch.arange(lanes)
for _ in range(300):  # leave the standing start
    with torch.no_grad():
        mu, sd = actor(norm(raw, lane_ids))
    act = torch.tanh(mu + sd * torch.randn(mu.shape, generator=gen)).numpy()
    for i, env in enumerate(envs):
        raw[i] = env.step(act[i])[0]
O, A, EPS, R, NO, QP, QV = [], [], [], [], [], [], []
for t in range(STEPS):
    obs = norm(raw, lane_ids)
    with torch.no_grad():
        mu, sd = actor(obs)
    eps = torch.randn(mu.shape, generator=gen)
    act = torch.tanh(mu + sd * eps)
    QP.append(np.stack([e.unwrapped.data.qpos.copy() for e in envs]))
    QV.append(np.stack([e.unwrapped.data.qvel.copy() for e in envs]))
    rew = np.zeros(lanes, dtype=np.float32)
    for i, env in enumerate(envs):
        raw[i], r, _, trunc, _ = env.step(act[i].numpy())
        rew[i] = r * args.reward_scale
        assert not trunc
    O.append(obs), A.append(act), EPS.append(eps), R.append(torch.as_tensor(rew)), NO.append(norm(raw, lane_ids))
obs, act, eps, rew, nobs = (torch.stack(x) for x in (O, A, EPS, R, NO))  # (T, lanes, ...)
with torch.no_grad():
    look1 = rew + gamma * soft_value(nobs.flatten(0, 1), v_noise).view(STEPS, lanes)
    mu_all, sd_all = actor(obs.flatten(0, 1))
train = slice(0, lanes - HOLD)
tr_obs, tr_act, tr_eps, tr_y = (x[:, train].flatten(0, 1) for x in (obs, act, eps, look1))
print(f"{path.split('/')[-1]}: train rows {len(tr_y)}, label mean {tr_y.mean():.2f} sd {tr_y.std():.3f}; "
      f"frozen critic rms(Q - L1) on train rows {(frozen.value(tr_obs, tr_act) - tr_y).square().mean().sqrt():.3f}")

# held-out states, K fresh actions each, exact L1 through the simulator
rng = np.random.default_rng(0)
n_eval = 384
pick_t = rng.integers(0, STEPS, n_eval)
pick_l = rng.integers(lanes - HOLD, lanes, n_eval)
ev_obs = obs[pick_t, pick_l]
with torch.no_grad():
    mu, sd = actor(ev_obs)
ev_eps = torch.randn(n_eval, K, act_dim, generator=gen)
ev_act = torch.tanh(mu[:, None] + sd[:, None] * ev_eps)
ev_y = torch.zeros(n_eval, K)
env = envs[0].unwrapped
start = time.perf_counter()
next_raw = np.zeros((n_eval, K, obs_dim))
ev_rew = np.zeros((n_eval, K), dtype=np.float32)
for j in range(n_eval):
    for k in range(K):
        env.set_state(QP[pick_t[j]][pick_l[j]], QV[pick_t[j]][pick_l[j]])
        o, r, _, _, _ = env.step(ev_act[j, k].numpy())
        next_raw[j, k], ev_rew[j, k] = o, r * args.reward_scale
with torch.no_grad():
    lanes_rep = torch.as_tensor(pick_l)[:, None].expand(-1, K).flatten()
    ev_y = torch.as_tensor(ev_rew) + gamma * soft_value(norm(next_raw.reshape(-1, obs_dim), lanes_rep), v_noise).view(n_eval, K)
# sanity: set_state branching reproduces the rollout's own transition
j = 0
env.set_state(QP[pick_t[j]][pick_l[j]], QV[pick_t[j]][pick_l[j]])
o, r, _, _, _ = env.step(act[pick_t[j], pick_l[j]].numpy())
print(f"branch check: |r - r_rollout| {abs(r * args.reward_scale - float(rew[pick_t[j], pick_l[j]])):.1e}, "
      f"|L1 - L1_rollout| {abs(float(r * args.reward_scale + gamma * soft_value(norm(o[None], torch.tensor([pick_l[j]])), v_noise)) - float(look1[pick_t[j], pick_l[j]])):.1e}")


class Whitened(torch.nn.Module):
    """Same critic, but the action enters as the behavior policy's standardized noise (atanh(a) - mu(s)) / sigma(s)."""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic
        self.hl = critic.hl

    def encode(self, o, a):
        mean, std = actor(o)
        pre = torch.atanh(a.clamp(-1 + 1e-6, 1 - 1e-6))
        return (pre - mean) / std

    def logits_of(self, o, a):
        return self.critic.logits(self.critic.features(o, self.encode(o, a)))

    def value(self, o, a):
        return self.hl.to_scalar(self.logits_of(o, a))


class Raw(Whitened):
    def encode(self, o, a):
        return a


def score(model, tag):
    with torch.no_grad():
        q = model.value(ev_obs[:, None].expand(-1, K, -1), ev_act)
    level = q.mean(1)
    keep = level > level.quantile(0.02)
    c = (q - q.mean(1, keepdim=True))[keep]
    t = (ev_y - ev_y.mean(1, keepdim=True))[keep]
    per = (c * t).sum(1) / ((c * c).sum(1) * (t * t).sum(1)).sqrt()
    return (f"{tag}: within-state corr {(c * t).sum() / ((c * c).sum() * (t * t).sum()).sqrt():+.3f} (per-state median {per.median():+.2f}), "
            f"amplitude sd(Q)/sd(L1) {(c.square().mean() / t.square().mean()).sqrt():.2f}, slope L1-on-Q {(c * t).sum() / (c * c).sum():+.2f}, "
            f"level rms {(q[keep] - ev_y[keep]).square().mean().sqrt():.3f}")


def fit(model, labels, updates, report, optimizer_state=None, tag=""):
    params = list(model.critic.parameters())
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)
    g = torch.Generator().manual_seed(3)
    print(score(model, f"  {tag} updates    0"), flush=True)
    for update in range(1, updates + 1):
        rows = torch.randint(0, len(labels), (args.minibatch_size,), generator=g)
        logits = model.logits_of(tr_obs[rows], tr_act[rows])
        loss = -(model.hl.project(labels[rows]) * logits.log_softmax(-1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        if update in report:
            with torch.no_grad():
                fit_rms = (model.value(tr_obs, tr_act) - labels).square().mean().sqrt()
            print(score(model, f"  {tag} updates {update:4d}") + f", train rms vs labels {fit_rms:.3f}", flush=True)


def loaded():
    critic = v60.Critic(obs_dim, act_dim, args, "cpu")
    critic.load_state_dict(ck["critic"])
    return critic


print(f"eval: {n_eval} held-out states x {K} fresh actions, within-state sd of L1 {(ev_y - ev_y.mean(1, keepdim=True)).square().mean().sqrt():.3f}")
noise = torch.randn(len(tr_y), generator=gen)
report = (128, 512, 1536)
print("A  trained checkpoint critic, continued on noise-free L1 labels (one action per state)")
fit(Raw(loaded()), tr_y, 1536, report, tag="A ")
print("N  same, labels + N(0, 0.4^2) (the trainer's fit residual at this step)")
fit(Raw(loaded()), tr_y + 0.4 * noise, 1536, report, tag="N ")
torch.manual_seed(0)
print("S  fresh critic from scratch, raw action input, noise-free labels")
fit(Raw(v60.Critic(obs_dim, act_dim, args, "cpu")), tr_y, 1536, report, tag="S ")
torch.manual_seed(0)
print("W  fresh critic from scratch, whitened-noise action input, noise-free labels")
fit(Whitened(v60.Critic(obs_dim, act_dim, args, "cpu")), tr_y, 1536, report, tag="W ")
torch.manual_seed(0)
print("SN fresh critic, raw action, noisy labels")
fit(Raw(v60.Critic(obs_dim, act_dim, args, "cpu")), tr_y + 0.4 * noise, 1536, report, tag="SN")
torch.manual_seed(0)
print("WN fresh critic, whitened action, noisy labels")
fit(Whitened(v60.Critic(obs_dim, act_dim, args, "cpu")), tr_y + 0.4 * noise, 1536, report, tag="WN")

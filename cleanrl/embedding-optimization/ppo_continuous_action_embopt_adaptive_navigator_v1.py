# Embedding Optimization: Adaptive Navigator v1 (embopt_adaptive_navigator_v1)
#
# Hypothesis: v25 plateaus because action-conditioned surrogate ascent does not
# learn a reusable path from the current state to an aspirational state. This probe
# removes every action-conditioned value. It learns an online LeJEPA representation,
# a SIGReg goal geometry, factual state-pair reward/step and arrival-time value,
# a multi-mode supervised navigator, and a supervised inverse controller.
#
# Factual state pairs are sampled at arbitrary durations from contiguous experience;
# there are no horizon bins. A goal persists until its learned remaining arrival
# time says it has arrived or a new goal offers materially better reward/step.
# There is no PPO, Q-learning, TD/bootstrap target, forced goal horizon, or EMA.
#
# The decisive failure test is navigator averaging: selected displacement error is
# logged against the zero-displacement baseline, together with norm shrinkage, cosine
# delivery, mode usage, support, extrapolation, and endogenous commitment telemetry.
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    wm_lr: float = 1e-4
    wm_weight_decay: float = 1e-3
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    max_grad_norm: float = 0.5

    latent_dim: int = 64
    goal_dim: int = 16
    hidden_dim: int = 256
    sigreg_coef: float = 0.09
    goal_sigreg_coef: float = 0.09
    sigreg_proj: int = 256
    sigreg_knots: int = 17
    sigreg_ref_n: int = 128

    replay_iters: int = 64
    pair_recent_iters: int = 8
    minibatch_size: int = 2048
    wm_updates: int = 16
    geometry_updates: int = 8
    pair_updates: int = 32
    command_updates: int = 8
    proposer_updates: int = 8
    navigator_modes: int = 4
    nav_balance_coef: float = 0.05

    goal_switch_margin: float = 0.1
    arrival_threshold: float = 1.5
    arrival_coef: float = 0.05
    expl_noise: float = 0.2
    warmup_steps: int = 10_000

    compile: bool = True
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden, out_dim, out_std=1.0):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


def mlp_ln(in_dim, hidden, out_dim, out_std=1.0):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer used by the LeJEPA lineage."""

    def __init__(self, knots=17, num_proj=256, ref_n=128):
        super().__init__()
        self.num_proj = num_proj
        self.ref_n = ref_n
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, x):
        projection = torch.randn(x.size(-1), self.num_proj, device=x.device, dtype=x.dtype)
        projection = projection / projection.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        x_t = (x @ projection).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return ((err @ self.weights) * self.ref_n).mean()


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        dz, dy, hidden = args.latent_dim, args.goal_dim, args.hidden_dim

        self.encoder = mlp_ln(obs_dim, hidden, dz)
        self.dyn = mlp_ln(dz + action_dim, hidden, dz, out_std=0.01)
        # Every goal-side caller passes detached z. LeJEPA learns only its own loss.
        self.goal_projector = mlp_ln(dz, hidden, dy)
        # Outputs factual segment reward/step and log factual segment duration.
        self.pair_value = mlp(3 * dy, hidden, 2)
        self.support_head = mlp(3 * dy, hidden, 1)
        self.navigator = mlp(3 * dy, hidden, args.navigator_modes * dy, out_std=0.01)
        self.inverse = mlp(dz + dy, hidden, action_dim, out_std=0.01)
        self.proposer = mlp(dy, hidden, dy)
        self.dy = dy
        self.navigator_modes = args.navigator_modes
        # Gauge-fix the whitened goal geometry without lagged/target parameters.
        self.register_buffer("goal_alignment", torch.eye(dy))
        # Avoid an untrained arrival head declaring every persistent goal complete.
        with torch.no_grad():
            self.pair_value[-1].bias[1] = math.log(32.0)

    def encode(self, obs):
        return self.encoder(obs)

    def forward_dyn(self, z, action):
        return z + self.dyn(torch.cat([z, action], -1))

    def goal_encode(self, z):
        return self.goal_projector(z.detach()) @ self.goal_alignment

    @staticmethod
    def pair_features(y, goal):
        return torch.cat([y, goal, goal - y], -1)

    def value(self, y, goal):
        output = self.pair_value(self.pair_features(y, goal))
        return output[..., 0], output[..., 1]

    def support(self, y, goal):
        return self.support_head(self.pair_features(y, goal)).squeeze(-1)

    def navigate(self, y, goal):
        shape = y.shape[:-1] + (self.navigator_modes, self.dy)
        return self.navigator(self.pair_features(y, goal)).reshape(shape)

    def act(self, z, desired_displacement):
        return torch.tanh(self.inverse(torch.cat([z, desired_displacement], -1)))

    def propose(self, y):
        raw = self.proposer(y)
        return (self.dy**0.5) * raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def factual_pair_targets(reward_prefix, episode_id, start, end, env):
    """Compute state-pair labels strictly from the executed trajectory."""
    steps = end - start
    reward_sum = reward_prefix[end, env] - reward_prefix[start, env]
    rate = reward_sum / steps.clamp_min(1).to(reward_sum.dtype)
    valid = (steps > 0) & (episode_id[start, env] == episode_id[end, env])
    return rate, steps.clamp_min(1).float().log(), valid.float()


def adaptive_goal_switch(current_rate, candidate_rate, current_log_steps, valid, margin, arrival_threshold):
    """Optimal-switching rule with no age or fixed-duration trigger."""
    arrived = current_log_steps.exp() <= arrival_threshold
    challenger = candidate_rate > current_rate + margin
    switch = (~valid) | arrived | challenger
    return switch, arrived & valid, challenger & valid & ~arrived


def navigator_loss(candidates, target, valid, balance_coef):
    """Best-of-many factual displacement loss plus mode-usage balancing."""
    per_mode = (candidates - target.unsqueeze(-2)).square().mean(-1)
    soft_assignment = torch.softmax(-per_mode, dim=-1)
    winner = per_mode.argmin(-1)
    chosen = per_mode.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
    denom = valid.sum().clamp_min(1.0)
    fit = (chosen * valid).sum() / denom
    usage = (soft_assignment * valid.unsqueeze(-1)).sum(0) / denom
    balance = (usage * usage.clamp_min(1e-8).log()).sum()
    return fit + balance_coef * balance, fit.detach(), winner.detach(), usage.detach()


def command_segment_targets(reward, done, switched):
    """Reward/step delivered by each exact command until switch, reset, or censoring."""
    steps, envs = reward.shape
    rate_target = torch.zeros_like(reward)
    duration = torch.zeros_like(reward)
    running_reward = torch.zeros(envs, device=reward.device, dtype=reward.dtype)
    running_steps = torch.zeros(envs, device=reward.device, dtype=reward.dtype)
    for step in range(steps - 1, -1, -1):
        boundary = done[step + 1].bool()
        if step + 1 < steps:
            boundary = boundary | switched[step + 1].bool()
        else:
            boundary = torch.ones_like(boundary)
        running_reward = torch.where(boundary, reward[step], reward[step] + running_reward)
        running_steps = torch.where(boundary, torch.ones_like(running_steps), running_steps + 1.0)
        rate_target[step] = running_reward / running_steps
        duration[step] = running_steps
    return rate_target, duration


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_envs * args.num_steps
    args.num_iterations = args.total_timesteps // args.batch_size
    assert 1 <= args.pair_recent_iters <= args.replay_iters
    assert args.navigator_modes >= 1
    assert args.cuda and torch.cuda.is_available(), "this experiment requires CUDA"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    action_dim = int(np.prod(envs.single_action_space.shape))
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    dz, dy = args.latent_dim, args.goal_dim
    T, E, R, mb = args.num_steps, args.num_envs, args.replay_iters, args.minibatch_size

    agent = Agent(envs, args).to(device)
    z_sigreg = SIGReg(args.sigreg_knots, args.sigreg_proj, args.sigreg_ref_n).to(device)
    y_sigreg = SIGReg(args.sigreg_knots, args.sigreg_proj, args.sigreg_ref_n).to(device)

    wm_params = list(agent.encoder.parameters()) + list(agent.dyn.parameters())
    geometry_params = list(agent.goal_projector.parameters())
    pair_params = (
        list(agent.pair_value.parameters())
        + list(agent.support_head.parameters())
        + list(agent.navigator.parameters())
        + list(agent.inverse.parameters())
    )
    proposer_params = list(agent.proposer.parameters())
    wm_opt = optim.AdamW(wm_params, lr=args.wm_lr, weight_decay=args.wm_weight_decay, eps=1e-5)
    geometry_opt = optim.Adam(geometry_params, lr=args.learning_rate, eps=1e-5)
    pair_opt = optim.Adam(pair_params, lr=args.learning_rate, eps=1e-5)
    proposer_opt = optim.Adam(proposer_params, lr=args.learning_rate, eps=1e-5)
    optimizers = (
        (wm_opt, args.wm_lr),
        (geometry_opt, args.learning_rate),
        (pair_opt, args.learning_rate),
        (proposer_opt, args.learning_rate),
    )

    def wm_loss_fn(obs, action, next_obs, valid):
        z = agent.encode(obs)
        next_z = agent.encode(next_obs)
        predicted = agent.forward_dyn(z, action)
        denom = valid.sum().clamp_min(1.0)
        pred_loss = (valid * (predicted - next_z).square().mean(-1)).sum() / denom
        sig_loss = z_sigreg(z)
        return pred_loss + args.sigreg_coef * sig_loss, pred_loss.detach(), sig_loss.detach()

    def geometry_loss_fn(z0, z1, rate_target, log_steps_target, valid):
        y0 = agent.goal_encode(z0)
        y1 = agent.goal_encode(z1)
        rate, log_steps = agent.value(y0, y1)
        denom = valid.sum().clamp_min(1.0)
        rate_loss = (
            nn.functional.smooth_l1_loss(rate, rate_target, reduction="none") * valid
        ).sum() / denom
        arrival_loss = (
            nn.functional.smooth_l1_loss(log_steps, log_steps_target, reduction="none") * valid
        ).sum() / denom
        goal_sig_loss = y_sigreg(torch.cat([y0, y1], 0))
        loss = rate_loss + arrival_loss + args.goal_sigreg_coef * goal_sig_loss
        return loss, rate_loss.detach(), arrival_loss.detach(), goal_sig_loss.detach()

    def pair_loss_fn(z0, z1, znext, action, rate_target, log_steps_target, valid):
        # Geometry was updated and gauge-fixed before this phase; dependent heads
        # train on a frozen global coordinate frame for the whole update.
        y0 = agent.goal_encode(z0).detach()
        y1 = agent.goal_encode(z1).detach()
        ynext = agent.goal_encode(znext).detach()
        rate, log_steps = agent.value(y0, y1)
        denom = valid.sum().clamp_min(1.0)
        rate_loss = (
            nn.functional.smooth_l1_loss(rate, rate_target, reduction="none") * valid
        ).sum() / denom
        arrival_loss = (
            nn.functional.smooth_l1_loss(log_steps, log_steps_target, reduction="none") * valid
        ).sum() / denom
        candidates = agent.navigate(y0, y1)
        # Targets are fixed factual coordinates. Navigator/inverse losses must not
        # reshape the goal geometry to make their own regression artificially easy.
        target_displacement = (ynext - y0).detach()
        nav_loss, nav_fit, _, _ = navigator_loss(
            candidates, target_displacement, valid, args.nav_balance_coef
        )
        predicted_action = agent.act(z0.detach(), target_displacement)
        inverse_loss = (
            (predicted_action - action).square().mean(-1) * valid
        ).sum() / denom
        positive = agent.support(y0, y1)
        negative_goal = agent.propose(y0.detach()).detach()
        negative = agent.support(y0, negative_goal)
        support_loss = (
            nn.functional.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction="none")
            + nn.functional.binary_cross_entropy_with_logits(
                negative, torch.zeros_like(negative), reduction="none"
            )
        )
        support_loss = (support_loss * valid).sum() / denom
        loss = (
            rate_loss
            + arrival_loss
            + nav_loss
            + inverse_loss
            + support_loss
        )
        return (
            loss,
            rate_loss.detach(),
            arrival_loss.detach(),
            nav_fit,
            inverse_loss.detach(),
            support_loss.detach(),
        )

    def command_loss_fn(command_y, command_goal, command_rate_target):
        command_rate, _ = agent.value(command_y, command_goal)
        return nn.functional.smooth_l1_loss(command_rate, command_rate_target)

    def proposer_loss_fn(y):
        goal = agent.propose(y)
        rate, log_steps = agent.value(y, goal)
        return -rate.mean(), rate.mean().detach(), log_steps.exp().mean().detach()

    def rollout_forward(obs, active_goal, goal_valid):
        z = agent.encode(obs)
        y = agent.goal_encode(z)
        candidate_goal = agent.propose(y)
        current_rate, current_log_steps = agent.value(y, active_goal)
        candidate_rate, candidate_log_steps = agent.value(y, candidate_goal)
        switch, arrived, challenger = adaptive_goal_switch(
            current_rate,
            candidate_rate,
            current_log_steps,
            goal_valid,
            args.goal_switch_margin,
            args.arrival_threshold,
        )
        goal = torch.where(switch.unsqueeze(-1), candidate_goal, active_goal)
        chosen_rate = torch.where(switch, candidate_rate, current_rate)
        chosen_log_steps = torch.where(switch, candidate_log_steps, current_log_steps)

        candidates = agent.navigate(y, goal)
        next_y = y.unsqueeze(1) + candidates
        expanded_goal = goal.unsqueeze(1).expand_as(next_y)
        mode_rate, mode_log_steps = agent.value(
            next_y.reshape(-1, dy), expanded_goal.reshape(-1, dy)
        )
        mode_score = mode_rate.reshape(E, args.navigator_modes) - args.arrival_coef * mode_log_steps.reshape(
            E, args.navigator_modes
        )
        mode = mode_score.argmax(-1)
        desired = candidates[torch.arange(E, device=device), mode]
        action = agent.act(z, desired)
        support = agent.support(y, goal).sigmoid()
        return (
            goal,
            action,
            y,
            desired,
            switch,
            arrived,
            challenger,
            chosen_rate,
            chosen_log_steps,
            support,
            mode,
        )

    if args.compile:
        wm_loss_fn = torch.compile(wm_loss_fn, mode=args.compile_mode, dynamic=False)
        geometry_loss_fn = torch.compile(geometry_loss_fn, mode=args.compile_mode, dynamic=False)
        pair_loss_fn = torch.compile(pair_loss_fn, mode=args.compile_mode, dynamic=False)
        command_loss_fn = torch.compile(command_loss_fn, mode=args.compile_mode, dynamic=False)
        proposer_loss_fn = torch.compile(proposer_loss_fn, mode=args.compile_mode, dynamic=False)
        rollout_forward = torch.compile(rollout_forward, mode=args.compile_mode, dynamic=False)
        print(f"[embopt_adaptive_navigator_v1] torch.compile(mode={args.compile_mode!r}, dynamic=False)")

    def mark_step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

    buf_obs = torch.zeros((R, T + 1, E, obs_dim), device=device)
    buf_act = torch.zeros((R, T, E, action_dim), device=device)
    buf_rew = torch.zeros((R, T, E), device=device)
    buf_done = torch.zeros((R, T + 1, E), device=device)
    buf_y = torch.zeros((R, T + 1, E, dy), device=device)
    buf_goal = torch.zeros((R, T, E, dy), device=device)
    buf_desired = torch.zeros((R, T, E, dy), device=device)
    buf_switch = torch.zeros((R, T, E), device=device)
    buf_arrived = torch.zeros((R, T, E), device=device)
    buf_challenger = torch.zeros((R, T, E), device=device)
    buf_goal_rate = torch.zeros((R, T, E), device=device)
    buf_goal_steps = torch.zeros((R, T, E), device=device)
    buf_goal_support = torch.zeros((R, T, E), device=device)
    buf_goal_age = torch.zeros((R, T, E), device=device)
    buf_mode = torch.zeros((R, T, E), dtype=torch.long, device=device)
    buf_filled, buf_ptr = 0, 0

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(E, device=device)
    active_goal = torch.zeros((E, dy), device=device)
    goal_valid = torch.zeros(E, dtype=torch.bool, device=device)
    goal_age = torch.zeros(E, device=device)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            for optimizer, base_lr in optimizers:
                optimizer.param_groups[0]["lr"] = fraction * base_lr

        slot = buf_ptr
        for step in range(T):
            buf_obs[slot, step] = next_obs
            buf_done[slot, step] = next_done
            goal_valid = goal_valid & ~next_done.bool()
            with torch.no_grad():
                mark_step()
                (
                    active_goal,
                    action,
                    current_y,
                    desired,
                    switched,
                    arrived,
                    challenger,
                    goal_rate,
                    goal_log_steps,
                    goal_support,
                    mode,
                ) = rollout_forward(next_obs, active_goal, goal_valid)
                goal_valid = torch.ones_like(goal_valid)
                goal_age = torch.where(switched, torch.zeros_like(goal_age), goal_age + 1.0)
                buf_y[slot, step] = current_y
                buf_goal[slot, step] = active_goal
                buf_desired[slot, step] = desired
                buf_switch[slot, step] = switched.float()
                buf_arrived[slot, step] = arrived.float()
                buf_challenger[slot, step] = challenger.float()
                buf_goal_rate[slot, step] = goal_rate
                buf_goal_steps[slot, step] = goal_log_steps.exp()
                buf_goal_support[slot, step] = goal_support
                buf_goal_age[slot, step] = goal_age
                buf_mode[slot, step] = mode
                action = action.clone()

            if global_step < args.warmup_steps:
                action = torch.empty((E, action_dim), device=device).uniform_(-1.0, 1.0)
            else:
                action = (action + args.expl_noise * torch.randn_like(action)).clamp(-1.0, 1.0)
            buf_act[slot, step] = action
            global_step += E

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            buf_rew[slot, step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        buf_obs[slot, T] = next_obs
        buf_done[slot, T] = next_done
        with torch.no_grad():
            buf_y[slot, T] = agent.goal_encode(agent.encode(next_obs))

        buf_filled = min(buf_filled + 1, R)
        fresh_slot = buf_ptr
        buf_ptr = (buf_ptr + 1) % R
        filled = buf_filled
        recent_count = min(args.pair_recent_iters, filled)
        recent_newest = torch.arange(fresh_slot, fresh_slot - recent_count, -1, device=device) % R
        recent_slots = recent_newest.flip(0)
        sequence_steps = recent_count * T
        frame_probe_obs = buf_obs[fresh_slot, :T].reshape(-1, obs_dim)[: min(512, T * E)]
        with torch.no_grad():
            frame_y_before = agent.goal_encode(agent.encode(frame_probe_obs))

        pair_valid_all = (buf_done[:filled, 1 : T + 1] == 0).float()
        wm_stats = []
        for _ in range(args.wm_updates):
            flat = torch.randint(0, filled * T * E, (mb,), device=device)
            replay_slot = flat // (T * E)
            replay_step = (flat // E) % T
            env_index = flat % E
            obs = buf_obs[replay_slot, replay_step, env_index]
            next_replay_obs = buf_obs[replay_slot, replay_step + 1, env_index]
            action = buf_act[replay_slot, replay_step, env_index]
            valid = pair_valid_all[replay_slot, replay_step, env_index]
            mark_step()
            loss, pred_loss, sig_loss = wm_loss_fn(obs, action, next_replay_obs, valid)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=wm_params)
            nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
            wm_opt.step()
            wm_stats.append((pred_loss.item(), sig_loss.item()))

        # Reconstruct a contiguous time axis, allowing factual pairs to span rollout
        # segments. Duration is sampled from endpoints, not from predefined horizons.
        obs_sequence = buf_obs[recent_slots, :T].reshape(sequence_steps, E, obs_dim)
        obs_sequence = torch.cat([obs_sequence, buf_obs[fresh_slot, T].unsqueeze(0)], 0)
        action_sequence = buf_act[recent_slots].reshape(sequence_steps, E, action_dim)
        reward_sequence = buf_rew[recent_slots].reshape(sequence_steps, E)
        done_sequence = buf_done[recent_slots, :T].reshape(sequence_steps, E)
        done_sequence = torch.cat([done_sequence, buf_done[fresh_slot, T].unsqueeze(0)], 0)
        episode_id = done_sequence.cumsum(0)
        reward_prefix = torch.cat(
            [torch.zeros((1, E), device=device), reward_sequence.cumsum(0)], 0
        )
        command_rate_pool, command_duration_pool = command_segment_targets(
            buf_rew[fresh_slot], buf_done[fresh_slot], buf_switch[fresh_slot]
        )
        command_y_pool = buf_y[fresh_slot, :T].reshape(-1, dy)
        command_goal_pool = buf_goal[fresh_slot].reshape(-1, dy)
        command_rate_pool = command_rate_pool.reshape(-1)
        with torch.no_grad():
            z_sequence = agent.encode(obs_sequence.reshape(-1, obs_dim)).reshape(
                sequence_steps + 1, E, dz
            )

        geometry_stats = []
        for _ in range(args.geometry_updates):
            first = torch.randint(0, sequence_steps + 1, (mb,), device=device)
            second = torch.randint(0, sequence_steps + 1, (mb,), device=device)
            pair_start = torch.minimum(first, second)
            pair_end = torch.maximum(first, second)
            equal = pair_start == pair_end
            pair_start = torch.where(equal & (pair_start == sequence_steps), pair_start - 1, pair_start)
            pair_end = torch.where(equal, pair_start + 1, pair_end)
            env_index = torch.randint(0, E, (mb,), device=device)
            rate_target, log_steps_target, valid = factual_pair_targets(
                reward_prefix, episode_id, pair_start, pair_end, env_index
            )
            z0 = z_sequence[pair_start, env_index]
            z1 = z_sequence[pair_end, env_index]
            mark_step()
            loss, rate_loss, arrival_loss, goal_sig_loss = geometry_loss_fn(
                z0, z1, rate_target, log_steps_target, valid
            )
            geometry_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=geometry_params)
            nn.utils.clip_grad_norm_(geometry_params, args.max_grad_norm)
            geometry_opt.step()
            geometry_stats.append((rate_loss.item(), arrival_loss.item(), goal_sig_loss.item()))

        with torch.no_grad():
            old_alignment = agent.goal_alignment.clone()
            frame_z_after = agent.encode(frame_probe_obs)
            frame_raw_after = agent.goal_projector(frame_z_after.detach())
            unaligned_after = frame_raw_after @ old_alignment
            projector_drift = (unaligned_after - frame_y_before).square().mean().item()
            # SIGReg leaves an orthogonal gauge freedom. Fix it before any dependent
            # head trains, so their inputs and displacement outputs stay global.
            u, _, vh = torch.linalg.svd(frame_raw_after.T @ frame_y_before)
            agent.goal_alignment.copy_(u @ vh)
            frame_y_after = frame_raw_after @ agent.goal_alignment
            goal_frame_residual = (frame_y_after - frame_y_before).square().mean().item()
            goal_frame_drift = (frame_y_after - frame_y_before).square().mean().item()

        pair_stats = []
        for _ in range(args.pair_updates):
            first = torch.randint(0, sequence_steps + 1, (mb,), device=device)
            second = torch.randint(0, sequence_steps + 1, (mb,), device=device)
            pair_start = torch.minimum(first, second)
            pair_end = torch.maximum(first, second)
            equal = pair_start == pair_end
            pair_start = torch.where(equal & (pair_start == sequence_steps), pair_start - 1, pair_start)
            pair_end = torch.where(equal, pair_start + 1, pair_end)
            env_index = torch.randint(0, E, (mb,), device=device)
            rate_target, log_steps_target, valid = factual_pair_targets(
                reward_prefix, episode_id, pair_start, pair_end, env_index
            )
            z0 = z_sequence[pair_start, env_index]
            z1 = z_sequence[pair_end, env_index]
            znext = z_sequence[pair_start + 1, env_index]
            action = action_sequence[pair_start, env_index]
            mark_step()
            result = pair_loss_fn(
                z0, z1, znext, action, rate_target, log_steps_target, valid
            )
            loss = result[0]
            pair_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=pair_params)
            nn.utils.clip_grad_norm_(pair_params, args.max_grad_norm)
            pair_opt.step()
            pair_stats.append(tuple(item.item() for item in result[1:6]))

        command_stats = []
        command_params = list(agent.pair_value.parameters())
        for _ in range(args.command_updates):
            command_index = torch.randint(0, T * E, (mb,), device=device)
            mark_step()
            command_loss = command_loss_fn(
                command_y_pool[command_index],
                command_goal_pool[command_index],
                command_rate_pool[command_index],
            )
            pair_opt.zero_grad(set_to_none=True)
            command_loss.backward(inputs=command_params)
            nn.utils.clip_grad_norm_(command_params, args.max_grad_norm)
            pair_opt.step()
            command_stats.append(command_loss.item())

        proposer_stats = []
        for _ in range(args.proposer_updates):
            flat = torch.randint(0, sequence_steps * E, (mb,), device=device)
            with torch.no_grad():
                z = z_sequence[:-1].reshape(-1, dz)[flat]
                y = agent.goal_encode(z)
            mark_step()
            loss, proposed_rate, proposed_steps = proposer_loss_fn(y)
            proposer_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=proposer_params)
            nn.utils.clip_grad_norm_(proposer_params, args.max_grad_norm)
            proposer_opt.step()
            proposer_stats.append((proposed_rate.item(), proposed_steps.item()))

        with torch.no_grad():
            # Fresh diagnostic pairs are sampled after all updates and never optimized.
            first = torch.randint(0, sequence_steps + 1, (mb,), device=device)
            second = torch.randint(0, sequence_steps + 1, (mb,), device=device)
            pair_start = torch.minimum(first, second)
            pair_end = torch.maximum(first, second)
            equal = pair_start == pair_end
            pair_start = torch.where(
                equal & (pair_start == sequence_steps), pair_start - 1, pair_start
            )
            pair_end = torch.where(equal, pair_start + 1, pair_end)
            env_index = torch.randint(0, E, (mb,), device=device)
            rate_target, log_steps_target, valid = factual_pair_targets(
                reward_prefix, episode_id, pair_start, pair_end, env_index
            )
            z0 = z_sequence[pair_start, env_index]
            z1 = z_sequence[pair_end, env_index]
            znext = z_sequence[pair_start + 1, env_index]
            action = action_sequence[pair_start, env_index]
            y0 = agent.goal_encode(z0)
            y1 = agent.goal_encode(z1)
            ynext = agent.goal_encode(znext)
            candidates = agent.navigate(y0, y1)
            target_displacement = ynext - y0
            per_mode = (candidates - target_displacement.unsqueeze(1)).square().mean(-1)
            winner = per_mode.argmin(-1)
            selected = candidates[torch.arange(mb, device=device), winner]
            denom = valid.sum().clamp_min(1.0)
            nav_mse = (per_mode.min(-1).values * valid).sum() / denom
            nav_zero_mse = (target_displacement.square().mean(-1) * valid).sum() / denom
            nav_zero_ratio = (nav_mse / nav_zero_mse.clamp_min(1e-8)).item()
            nav_cosine = (
                nn.functional.cosine_similarity(selected, target_displacement, dim=-1) * valid
            ).sum() / denom
            nav_norm_ratio = (
                selected.norm(dim=-1) / target_displacement.norm(dim=-1).clamp_min(1e-6) * valid
            ).sum() / denom
            nav_mode_spread = candidates.std(dim=1).square().mean().sqrt().item()
            winner_hist = torch.bincount(winner[valid.bool()], minlength=args.navigator_modes).float()
            winner_prob = winner_hist / winner_hist.sum().clamp_min(1.0)
            winner_entropy = (
                -(winner_prob * winner_prob.clamp_min(1e-8).log()).sum()
                / math.log(args.navigator_modes)
                if args.navigator_modes > 1
                else torch.zeros((), device=device)
            ).item()
            inverse_action = agent.act(z0, target_displacement)
            inverse_action_mse = (
                (inverse_action - action).square().mean(-1) * valid
            ).sum().div(denom).item()
            predicted_rate, predicted_log_steps = agent.value(y0, y1)
            rate_abs_error = (
                (predicted_rate - rate_target).abs() * valid
            ).sum().div(denom).item()
            arrival_ratio = (
                (predicted_log_steps.exp() / log_steps_target.exp() - 1.0).abs() * valid
            ).sum().div(denom).item()
            factual_support = (
                agent.support(y0, y1).sigmoid() * valid
            ).sum().div(denom).item()

            rollout_y = buf_y[fresh_slot]
            rollout_goal = buf_goal[fresh_slot]
            actual_displacement = rollout_y[1:] - rollout_y[:-1]
            desired_displacement = buf_desired[fresh_slot]
            delivery_valid = (buf_done[fresh_slot, 1:] == 0).float()
            delivery_denom = delivery_valid.sum().clamp_min(1.0)
            desired_cosine = (
                nn.functional.cosine_similarity(
                    desired_displacement, actual_displacement, dim=-1
                )
                * delivery_valid
            ).sum().div(delivery_denom).item()
            desired_norm_ratio = ((
                actual_displacement.norm(dim=-1)
                / desired_displacement.norm(dim=-1).clamp_min(1e-6)
            ) * delivery_valid).sum().div(delivery_denom).item()
            before_distance = (rollout_goal - rollout_y[:-1]).square().mean(-1)
            after_distance = (rollout_goal - rollout_y[1:]).square().mean(-1)
            goal_distance = (before_distance * delivery_valid).sum().div(delivery_denom).item()
            goal_progress = (
                (before_distance - after_distance) * delivery_valid
            ).sum().div(delivery_denom).item()

            diag_count = min(512, sequence_steps * E)
            diag_z = z_sequence[:-1].reshape(-1, dz)[:diag_count]
            diag_y = agent.goal_encode(diag_z)
            proposed_goal = agent.propose(diag_y)
            proposal_support = agent.support(diag_y, proposed_goal).sigmoid().mean().item()
            proposal_distance = torch.cdist(
                proposed_goal[: min(256, diag_count)], diag_y
            ).min(-1).values.mean().item()
            real_permutation = torch.randperm(diag_count, device=device)
            real_rate, _ = agent.value(diag_y, diag_y[real_permutation])
            proposal_rate, _ = agent.value(diag_y, proposed_goal)
            proposal_excess = (proposal_rate.mean() - real_rate.max()).item()

            switch_rate = buf_switch[fresh_slot].mean().item()
            arrived_switch_rate = buf_arrived[fresh_slot].mean().item()
            challenger_switch_rate = buf_challenger[fresh_slot].mean().item()
            commitment_mean = buf_goal_age[fresh_slot].mean().item()
            commitment_max = buf_goal_age[fresh_slot].max().item()
            predicted_goal_steps = buf_goal_steps[fresh_slot].mean().item()
            goal_support = buf_goal_support[fresh_slot].mean().item()
            goal_promise = buf_goal_rate[fresh_slot].mean().item()
            goal_delivery = buf_rew[fresh_slot].mean().item()
            command_rate_prediction, _ = agent.value(command_y_pool, command_goal_pool)
            command_rate_abs_error = (
                command_rate_prediction - command_rate_pool
            ).abs().mean().item()
            command_duration_mean = command_duration_pool.mean().item()
            control_cost = (0.1 * buf_act[fresh_slot].square().sum(-1)).mean().item()

            z_flat = z_sequence[:-1].reshape(-1, dz)
            z_centered = z_flat - z_flat.mean(0)
            z_eigenvalues = torch.linalg.eigvalsh(
                z_centered.T @ z_centered / (z_flat.shape[0] - 1)
            )
            z_effective_rank = (
                z_eigenvalues.sum().square() / z_eigenvalues.square().sum().clamp_min(1e-12)
            ).item()
            y_flat = agent.goal_encode(z_flat)
            y_centered = y_flat - y_flat.mean(0)
            y_eigenvalues = torch.linalg.eigvalsh(
                y_centered.T @ y_centered / (y_flat.shape[0] - 1)
            )
            y_effective_rank = (
                y_eigenvalues.sum().square() / y_eigenvalues.square().sum().clamp_min(1e-12)
            ).item()

        wm_mean = np.mean(wm_stats, axis=0)
        geometry_mean = np.mean(geometry_stats, axis=0)
        pair_mean = np.mean(pair_stats, axis=0)
        proposer_mean = np.mean(proposer_stats, axis=0)
        writer.add_scalar("charts/learning_rate", pair_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/wm_pred", wm_mean[0], global_step)
        writer.add_scalar("losses/wm_sigreg", wm_mean[1], global_step)
        writer.add_scalar("losses/pair_rate", pair_mean[0], global_step)
        writer.add_scalar("losses/pair_arrival", pair_mean[1], global_step)
        writer.add_scalar("losses/navigator", pair_mean[2], global_step)
        writer.add_scalar("losses/inverse", pair_mean[3], global_step)
        writer.add_scalar("losses/support", pair_mean[4], global_step)
        writer.add_scalar("losses/geometry_rate", geometry_mean[0], global_step)
        writer.add_scalar("losses/geometry_arrival", geometry_mean[1], global_step)
        writer.add_scalar("losses/goal_sigreg", geometry_mean[2], global_step)
        writer.add_scalar("losses/command_rate", np.mean(command_stats), global_step)
        writer.add_scalar("navigator/error_vs_zero", nav_zero_ratio, global_step)
        writer.add_scalar("navigator/cosine", nav_cosine.item(), global_step)
        writer.add_scalar("navigator/norm_ratio", nav_norm_ratio.item(), global_step)
        writer.add_scalar("navigator/mode_spread", nav_mode_spread, global_step)
        writer.add_scalar("navigator/winner_entropy", winner_entropy, global_step)
        writer.add_scalar("navigator/inverse_action_mse", inverse_action_mse, global_step)
        writer.add_scalar("delivery/desired_actual_cosine", desired_cosine, global_step)
        writer.add_scalar("delivery/actual_desired_norm_ratio", desired_norm_ratio, global_step)
        writer.add_scalar("delivery/goal_distance", goal_distance, global_step)
        writer.add_scalar("delivery/goal_progress", goal_progress, global_step)
        writer.add_scalar("value/rate_abs_error", rate_abs_error, global_step)
        writer.add_scalar("value/arrival_relative_error", arrival_ratio, global_step)
        writer.add_scalar("value/command_rate_abs_error", command_rate_abs_error, global_step)
        writer.add_scalar("value/command_duration_mean", command_duration_mean, global_step)
        writer.add_scalar("value/proposer_rate", proposer_mean[0], global_step)
        writer.add_scalar("value/proposer_steps", proposer_mean[1], global_step)
        writer.add_scalar("support/factual_probability", factual_support, global_step)
        writer.add_scalar("support/proposal_probability", proposal_support, global_step)
        writer.add_scalar("support/proposal_nearest_real_distance", proposal_distance, global_step)
        writer.add_scalar("support/proposal_rate_excess", proposal_excess, global_step)
        writer.add_scalar("commitment/switch_rate", switch_rate, global_step)
        writer.add_scalar("commitment/arrived_switch_rate", arrived_switch_rate, global_step)
        writer.add_scalar("commitment/challenger_switch_rate", challenger_switch_rate, global_step)
        writer.add_scalar("commitment/age_mean", commitment_mean, global_step)
        writer.add_scalar("commitment/age_max", commitment_max, global_step)
        writer.add_scalar("commitment/predicted_steps", predicted_goal_steps, global_step)
        writer.add_scalar("commitment/goal_support", goal_support, global_step)
        writer.add_scalar("commitment/goal_promise", goal_promise, global_step)
        writer.add_scalar("commitment/goal_delivery", goal_delivery, global_step)
        writer.add_scalar("diagnostics/goal_frame_drift", goal_frame_drift, global_step)
        writer.add_scalar("diagnostics/goal_frame_unaligned_drift", projector_drift, global_step)
        writer.add_scalar("diagnostics/goal_frame_alignment_residual", goal_frame_residual, global_step)
        writer.add_scalar("diagnostics/z_effective_rank", z_effective_rank, global_step)
        writer.add_scalar("diagnostics/y_effective_rank", y_effective_rank, global_step)
        writer.add_scalar("diagnostics/control_cost", control_cost, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"iter={iteration} SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()

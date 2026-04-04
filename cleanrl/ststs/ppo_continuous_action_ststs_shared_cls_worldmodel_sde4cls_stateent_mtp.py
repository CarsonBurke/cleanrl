# PPO + WM + SDE + state entropy + register tokens + causal temporal + multi-token pred.
#
# Fork of worldmodel_sde4cls_stateent with D4-inspired improvements:
# - 8 register tokens in spatial attention
# - Causal temporal attention
# - Multi-token prediction: state_pred CLS predicts next K=4 latents as Gaussians
# 6 CLS tokens: actor, critic, dynamics, SDE, state_pred + 8 registers
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


EMBED_DIM = 32
NUM_HEADS = 4
FFN_MULT = 2
CONTEXT_LEN = 5
NUM_SPATIAL_BLOCKS = 3
NUM_TEMPORAL_BLOCKS = 2
NUM_CLS_TOKENS = 5
N_REGISTER = 8
LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
IMAGINATION_HORIZON = 15
WM_COEF = 1.0
IMAGINE_COEF = 0.1
IMAGINE_CRITIC_COEF = 0.5
N_IMAGINE_SEEDS = 256
STATE_ENT_COEF = 0.01
STATE_PRED_LOSS_COEF = 0.1
MTP_K = 4


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
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def layer_init(layer, std=None, bias_const=0.0):
    fan_in = layer.weight.shape[1]
    if std is None: std = 1.0 / fan_in**0.5
    nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + self.eps) * self.weight


def build_rope_cache(seq_len, head_dim, device):
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), theta)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_mult=2, init_scale=1.0):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = dim // num_heads
        self.attn_pre_norm = RMSNorm(dim); self.attn_post_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False); self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False); self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim); self.k_norm = RMSNorm(self.head_dim)
        self.ffn_pre_norm = RMSNorm(dim); self.ffn_post_norm = RMSNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_value = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=False)
        for m in [self.q_proj, self.k_proj, self.v_proj, self.ffn_gate, self.ffn_value]:
            nn.init.trunc_normal_(m.weight, std=(1.0/m.weight.shape[1]**0.5))
        nn.init.trunc_normal_(self.out_proj.weight, std=0.1*init_scale/self.out_proj.weight.shape[1]**0.5)
        nn.init.trunc_normal_(self.ffn_out.weight, std=init_scale/self.ffn_out.weight.shape[1]**0.5)

    def forward(self, x, rope_cos=None, rope_sin=None, is_causal=False):
        b, s, w = x.shape
        h = self.attn_pre_norm(x)
        q = self.q_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin); k = apply_rope(k, rope_cos, rope_sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        x = x + self.attn_post_norm(self.out_proj(attn.transpose(1, 2).reshape(b, s, w)))
        h = self.ffn_pre_norm(x)
        x = x + self.ffn_post_norm(self.ffn_out(F.silu(self.ffn_gate(h)) * self.ffn_value(h)))
        return x


class STSTSCLSBackbone(nn.Module):
    def __init__(self, obs_dim, context_len):
        super().__init__()
        self.obs_dim = obs_dim; self.context_len = context_len
        base = obs_dim + N_REGISTER
        self.actor_cls_index = base
        self.critic_cls_index = base + 1
        self.dynamics_cls_index = base + 2
        self.sde_cls_index = base + 3
        self.state_pred_cls_index = base + 4

        self.value_proj = layer_init(nn.Linear(1, EMBED_DIM), std=1.0)
        self.input_norm = RMSNorm(EMBED_DIM)
        self.dim_id_embed = nn.Embedding(obs_dim, EMBED_DIM)
        self.register_buffer("dim_indices", torch.arange(obs_dim))
        cls_std = 1.0 / EMBED_DIM**0.5
        self.actor_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.critic_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.dynamics_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.sde_cls = nn.Parameter(torch.empty(EMBED_DIM))
        self.state_pred_cls = nn.Parameter(torch.empty(EMBED_DIM))
        for p in [self.actor_cls, self.critic_cls, self.dynamics_cls, self.sde_cls, self.state_pred_cls]:
            nn.init.trunc_normal_(p, std=cls_std, a=-2*cls_std, b=2*cls_std)
        self.register_tokens = nn.Parameter(torch.randn(N_REGISTER, EMBED_DIM) * 1e-2)

        init_scale = 1.0 / (2 * (NUM_SPATIAL_BLOCKS + NUM_TEMPORAL_BLOCKS)) ** 0.5
        self.s_blocks = nn.ModuleList([SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(NUM_SPATIAL_BLOCKS)])
        self.t_blocks = nn.ModuleList([SelfAttentionBlock(EMBED_DIM, NUM_HEADS, FFN_MULT, init_scale) for _ in range(NUM_TEMPORAL_BLOCKS)])
        self.final_norm = RMSNorm(EMBED_DIM)
        tc, ts = build_rope_cache(context_len, EMBED_DIM // NUM_HEADS, torch.device("cpu"))
        self.register_buffer("temporal_cos", tc); self.register_buffer("temporal_sin", ts)

    def _spatial(self, tokens, block):
        b, t, s, w = tokens.shape
        return block(tokens.reshape(b*t, s, w)).reshape(b, t, s, w)
    def _temporal(self, tokens, block):
        b, t, s, w = tokens.shape
        x = tokens.permute(0,2,1,3).reshape(b*s, t, w)
        x = block(x, rope_cos=self.temporal_cos, rope_sin=self.temporal_sin, is_causal=True)
        return x.reshape(b, s, t, w).permute(0,2,1,3)

    def forward(self, obs_seq):
        batch, time_steps, _ = obs_seq.shape
        obs_tokens = self.value_proj(obs_seq.unsqueeze(-1)) + self.dim_id_embed(self.dim_indices).view(1,1,self.obs_dim,EMBED_DIM)
        reg = self.register_tokens.view(1,1,N_REGISTER,EMBED_DIM).expand(batch, time_steps, -1, -1)
        cls = torch.stack([self.actor_cls, self.critic_cls, self.dynamics_cls, self.sde_cls, self.state_pred_cls])
        cls = cls.view(1,1,NUM_CLS_TOKENS,EMBED_DIM).expand(batch, time_steps, -1, -1)
        tokens = self.input_norm(torch.cat([obs_tokens, reg, cls], dim=2))
        tokens = self._spatial(tokens, self.s_blocks[0]); tokens = self._temporal(tokens, self.t_blocks[0])
        tokens = self._spatial(tokens, self.s_blocks[1]); tokens = self._temporal(tokens, self.t_blocks[1])
        tokens = self._spatial(tokens, self.s_blocks[2]); tokens = self.final_norm(tokens)
        t = tokens[:, -1]
        return t[:, self.actor_cls_index], t[:, self.critic_cls_index], t[:, self.dynamics_cls_index], t[:, self.sde_cls_index], t[:, self.state_pred_cls_index]


class Agent(nn.Module):
    def __init__(self, envs, num_envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim; self.context_len = CONTEXT_LEN
        self.backbone = STSTSCLSBackbone(obs_dim, self.context_len)
        self.actor_mean = layer_init(nn.Linear(EMBED_DIM, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(EMBED_DIM, 1), std=1.0)
        self.log_std_param = nn.Parameter(torch.zeros(EMBED_DIM, action_dim))
        self.to_state_pred = nn.Sequential(RMSNorm(EMBED_DIM), nn.Linear(EMBED_DIM, MTP_K * EMBED_DIM * 2))
        self.transition = nn.Sequential(nn.Linear(EMBED_DIM+action_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, EMBED_DIM))
        self.reward_head = nn.Sequential(nn.Linear(EMBED_DIM+action_dim, 128), nn.SiLU(), nn.Linear(128, 1))
        self.continue_head = nn.Sequential(nn.Linear(EMBED_DIM+action_dim, 128), nn.SiLU(), nn.Linear(128, 1))
        self.register_buffer("obs_history", torch.zeros(num_envs, self.context_len, obs_dim))

    def reset_history(self, env_mask=None):
        if env_mask is None: self.obs_history.zero_()
        else: self.obs_history[env_mask] = 0.0
    def update_history(self, obs):
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs.unsqueeze(1)], dim=1)
    def _encode(self, obs_seq): return self.backbone(obs_seq)
    def _get_action_std(self, sde):
        sde = torch.tanh(sde)
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return (sde.pow(2) @ log_std.exp().pow(2) + 1e-6).sqrt()
    def _action_std_fixed(self):
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return log_std.exp().pow(2).mean(0).sqrt()
    def get_state_entropy_bonus(self, sp):
        pred = self.to_state_pred(sp).view(-1, MTP_K, EMBED_DIM, 2)
        return pred[..., 1].mean(dim=(-1, -2)) * STATE_ENT_COEF
    def get_value(self, obs_seq):
        _, c, _, _, _ = self._encode(obs_seq); return self.critic(c)

    def get_action_and_value(self, obs_seq, action=None):
        a, c, d, s, sp = self._encode(obs_seq)
        mean = self.actor_mean(a); std = self._get_action_std(s)
        probs = Normal(mean, std)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(c), d, sp

    def get_all_for_update(self, obs_seq, action):
        a, c, d, s, sp = self._encode(obs_seq)
        mean = self.actor_mean(a); std = self._get_action_std(s)
        probs = Normal(mean, std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(c), d, sp

    def imagine(self, z_start, horizon, gamma, gae_lambda):
        action_std = self._action_std_fixed(); z = z_start
        rewards, values, continues = [], [], []
        for _ in range(horizon):
            a = Normal(self.actor_mean(z), action_std).rsample()
            za = torch.cat([z, a], dim=-1)
            rewards.append(self.reward_head(za).squeeze(-1))
            continues.append(self.continue_head(za).sigmoid().squeeze(-1))
            values.append(self.critic(z).squeeze(-1))
            z = self.transition(za)
        values.append(self.critic(z).squeeze(-1))
        ret = values[-1]; lambda_returns = []
        for t in reversed(range(horizon)):
            ret = rewards[t] + gamma * continues[t] * (gae_lambda * ret + (1-gae_lambda) * values[t+1])
            lambda_returns.append(ret)
        lambda_returns.reverse()
        return lambda_returns, values[:-1]


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])

    agent = Agent(envs, args.num_envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    next_obs_seqs = torch.zeros((args.num_steps, args.num_envs, agent.context_len, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0; start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)
    agent.reset_history(); agent.update_history(next_obs)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (iteration-1.0)/args.num_iterations) * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_seqs[step] = agent.obs_history.clone(); dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, _, sp = agent.get_action_and_value(agent.obs_history)
                values[step] = value.flatten()
                ent_bonus = agent.get_state_entropy_bonus(sp)
            actions[step] = action; logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device).view(-1) + ent_bonus
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
            if next_done.any(): agent.reset_history(next_done.bool())
            agent.update_history(next_obs)
            next_obs_seqs[step] = agent.obs_history.clone()
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(agent.obs_history).reshape(1, -1)
            advantages = torch.zeros_like(rewards); lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                nnt = (1.0-next_done) if t == args.num_steps-1 else (1.0-dones[t+1])
                nv = next_value if t == args.num_steps-1 else values[t+1]
                delta = rewards[t] + args.gamma * nv * nnt - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nnt * lastgaelam
            returns = advantages + values

        # Multi-token targets
        mtp_targets = []; mtp_valid = []
        cum_valid = torch.ones(args.num_steps, args.num_envs, device=device)
        for k in range(1, MTP_K + 1):
            if k == 1:
                cum_valid = cum_valid.clone(); cum_valid[:-1] *= (1.0 - dones[1:]); cum_valid[-1] = 0.0
            else:
                sd = torch.zeros(args.num_steps, args.num_envs, device=device)
                if k <= args.num_steps: sd[:-k] = dones[k:]
                cum_valid = cum_valid * (1.0 - sd)
            sv = torch.zeros(args.num_steps, args.num_envs, device=device)
            if k < args.num_steps: sv[:args.num_steps-k] = 1.0
            valid = cum_valid * sv
            target = torch.zeros_like(obs_seqs)
            if k < args.num_steps: target[:args.num_steps-k] = obs_seqs[k:]
            mtp_targets.append(target.reshape(-1, agent.context_len, obs_dim))
            mtp_valid.append(valid.reshape(-1))

        b_obs_seqs = obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_next_obs_seqs = next_obs_seqs.reshape(-1, agent.context_len, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1); b_returns = returns.reshape(-1)
        b_values = values.reshape(-1); b_rewards = rewards.reshape(-1)
        wm_valid = torch.ones(args.num_steps, args.num_envs, device=device)
        wm_valid[:-1] = 1.0 - dones[1:]
        b_wm_valid = wm_valid.reshape(-1)

        b_inds = np.arange(args.batch_size); clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start+args.minibatch_size]
                newlp, ent, nv, dyn, sp = agent.get_all_for_update(b_obs_seqs[mb], b_actions[mb])
                lr = newlp - b_logprobs[mb]; ratio = lr.exp()
                with torch.no_grad():
                    approx_kl = ((ratio-1.0)-lr).mean()
                    clipfracs += [((ratio-1.0).abs() > args.clip_coef).float().mean().item()]
                mb_adv = b_advantages[mb]
                if args.norm_adv: mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv*ratio, -mb_adv*ratio.clamp(1-args.clip_coef, 1+args.clip_coef)).mean()
                nv = nv.view(-1)
                if args.clip_vloss:
                    vc = b_values[mb] + (nv-b_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                    v_loss = 0.5*torch.max((nv-b_returns[mb])**2, (vc-b_returns[mb])**2).mean()
                else: v_loss = 0.5*((nv-b_returns[mb])**2).mean()
                entropy_loss = ent.mean()

                # WM loss fresh
                with torch.no_grad():
                    _, _, next_dyn, _, _ = agent._encode(b_next_obs_seqs[mb])
                za = torch.cat([dyn, b_actions[mb]], dim=-1)
                mv = b_wm_valid[mb]
                t_loss = (((agent.transition(za) - next_dyn)**2).mean(-1)*mv).sum()/(mv.sum()+1e-8)
                r_loss = ((agent.reward_head(za).squeeze(-1) - b_rewards[mb])**2*mv).sum()/(mv.sum()+1e-8)
                c_loss = F.binary_cross_entropy_with_logits(agent.continue_head(za).squeeze(-1), mv, reduction="mean")
                wm_loss = t_loss + r_loss + c_loss

                # MTP state pred loss
                pred = agent.to_state_pred(sp).view(-1, MTP_K, EMBED_DIM, 2)
                sp_loss = torch.tensor(0.0, device=device)
                for ki in range(MTP_K):
                    with torch.no_grad():
                        _, _, tgt_dyn, _, _ = agent._encode(mtp_targets[ki][mb])
                    pm, pv = pred[:, ki, :, 0], pred[:, ki, :, 1].exp()
                    vld = mtp_valid[ki][mb]
                    raw = F.gaussian_nll_loss(pm, tgt_dyn, var=pv, reduction='none')
                    sp_loss = sp_loss + (raw.mean(-1)*vld).sum()/(vld.sum()+1e-8)
                sp_loss = sp_loss / MTP_K

                loss = pg_loss - args.ent_coef*entropy_loss + args.vf_coef*v_loss + WM_COEF*wm_loss + STATE_PRED_LOSS_COEF*sp_loss
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl: break

        # Imagination
        n_seeds = min(N_IMAGINE_SEEDS, args.batch_size)
        with torch.no_grad():
            _, _, zs, _, _ = agent._encode(b_obs_seqs[torch.randperm(args.batch_size, device=device)[:n_seeds]])
        lrs, ivs = agent.imagine(zs.detach(), IMAGINATION_HORIZON, args.gamma, args.gae_lambda)
        im_actor = -torch.stack(lrs).mean()
        im_critic = sum(((ivs[t]-lrs[t].detach())**2).mean() for t in range(IMAGINATION_HORIZON)) / IMAGINATION_HORIZON
        im_loss = IMAGINE_COEF*im_actor + IMAGINE_CRITIC_COEF*im_critic
        optimizer.zero_grad(); im_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); optimizer.step()

        # Log
        yp, yt = b_values.cpu().numpy(), b_returns.cpu().numpy()
        ev = np.nan if np.var(yt)==0 else 1-np.var(yt-yp)/np.var(yt)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", ev, global_step)
        writer.add_scalar("worldmodel/total_loss", wm_loss.item(), global_step)
        writer.add_scalar("state_pred/loss", sp_loss.item(), global_step)
        writer.add_scalar("imagination/actor_loss", im_actor.item(), global_step)
        with torch.no_grad():
            writer.add_scalar("imagination/mean_return", torch.stack(lrs).mean().item(), global_step)
            writer.add_scalar("policy/action_std_mean", agent._action_std_fixed().mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step/(time.time()-start_time)), global_step)

    if args.save_model:
        torch.save(agent.state_dict(), f"runs/{run_name}/{args.exp_name}.cleanrl_model")
    envs.close(); writer.close()

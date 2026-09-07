"""How large is the covariance term of a per-unit plasticity gate, in real PPO?

THE ALGEBRA THAT DEFINES THE TARGET. For unit `h` with layer input `x_t`,
preactivation gradient `d_th` and gate `c_th`, the gated row update splits
exactly into

    sum_t c_th d_th x_t  =  cbar_h * sum_t d_th x_t  +  sum_t (c_th - cbar_h) p_th
                            \\________ LR term ______/    \\____ COVARIANCE term ___/

with `p_th = d_th x_t` and `cbar_h = mean_t c_th`. The first term is a per-unit
learning-rate change, which is the confound that faked four earlier results in
this family. ALL of the mechanism's real content is the second term. The figure
of merit is therefore

    ratio_h = || sum_t (c_th - cbar_h) p_th ||  /  || cbar_h * sum_t d_th x_t ||

i.e. the mechanism's maximum possible relative effect on the step. Gate variance
alone buys nothing: if `c` is uncorrelated with `p` the covariance term grows
like sqrt(B) while the LR term grows like B, so the ratio decays as 1/sqrt(B)
and vanishes at PPO's batch. If the covariance is systematic the ratio is flat
in B. That contrast is measurement 4 and it is the headline.

WHAT IS MEASURED, on real clipped-PPO gradients from real HalfCheetah rollouts,
at three training stages, with paired replicates for error bars.

1. RATIO, per unit and layer, for every gate arm below.
2. ORACLE CEILING. The held-out first-order loss reduction is LINEAR in the
   gate, `R(c) = sum_th c_th <p_th, ghat_h>`, so each ceiling has a closed form
   and no fitting is involved:
     ungated     c = 1
     free        c_th = 1[<p_th, ghat_h> > 0]      -- unrestricted upper bound
     state       c_th = w_h(bin of that unit's OWN preactivation), w optimal;
                 the gap to `free` is how much of the available covariance is
                 legible from a unit's own state at all
     shared      c_th = c_t, one decision for every unit -- the control that
                 says whether per-unit autonomy is needed
     shuffled    the `state` gate with its bin->weight map permuted per unit:
                 matched mean, matched dispersion, correspondence destroyed
   Gates are in [0, 1]: a plasticity rule can suppress a sample for a unit but
   never flip its sign. Scores are COSINES with the target gradient, never step
   sizes, so nothing can be won by taking a bigger step. A gate fitted against
   one minibatch's gradient is also scored on a THIRD, untouched minibatch; the
   held-out column is the honest one and the in-sample column is the ceiling.
3. TANH-SLOPE PROJECTION. An earlier harvest found that ~89% of the apparent
   own-state structure was the unit's own tanh derivative `1 - a^2`, i.e. the
   network re-applying its own Jacobian. The `state_orth` arm removes exactly
   that: the per-bin usefulness pattern is projected orthogonally to the per-bin
   mean slope before the gate is chosen, so whatever it retains is genuinely
   novel own-state content.
4. BATCH SCALING of the ratio at 512 / 4096 / 32768 samples, against the
   1/sqrt(B) line that uncorrelated gate variance would follow.

Diagnostic only: trains nothing that is scored and submits nothing.

    .venv/bin/python cleanrl/plasticity/ppo_batch_conflict.py
    .venv/bin/python cleanrl/plasticity/ppo_batch_conflict.py --probe-sizes 1024
"""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta

from cleanrl.plasticity.ppo_continuous_action_precision_v1 import Agent
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

ARMS = ("ungated", "free", "state", "state_orth", "shared", "shuffled")


@dataclass
class Args:
    env_id: str = "HalfCheetah-v4"
    env_threads: int = 4
    total_steps: int = 1_500_000
    """environment steps of real training to measure across"""
    num_envs: int = 16
    num_steps: int = 2048
    """rollout length; num_envs * num_steps is the benchmark recipe's batch"""
    minibatch_size: int = 1024
    """the recipe's optimizer-step batch, used for the training the probe watches"""
    probe_sizes: str = "512,4096,32768"
    """probe minibatch sizes for the batch-scaling test. The recipe's optimizer
    step actually sees 1024 samples; 32768 is the full rollout batch"""
    bins: int = 8
    """resolution of the state-readable gate: equal-count bins of a unit's own
    preactivation, so the `state` arm is the best gate at that resolution"""
    replicates: int = 4
    """independent probe triples per stage and size, for paired error bars"""
    stages: int = 3
    """report windows across training"""
    update_epochs: int = 10
    learning_rate: float = 8.1e-4
    seed: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True


class Probe(nn.Module):
    """The baseline Agent with every hidden layer's INPUT and PREACTIVATION exposed."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.trunks, self.names = [], []
        for trunk_name, trunk in (("actor", agent.actor), ("critic", agent.critic)):
            modules = list(trunk)
            linears = [i for i, module in enumerate(modules) if isinstance(module, nn.Linear)]
            tracked = linears[:-1]
            self.trunks.append((modules, set(tracked)))
            self.names.extend(f"{trunk_name}.{index}" for index in tracked)

    def forward(self, observations):
        inputs, preacts, outputs = [], [], []
        for modules, tracked in self.trunks:
            activations = observations
            for index, module in enumerate(modules):
                if index in tracked:
                    inputs.append(activations)
                activations = module(activations)
                if index in tracked:
                    activations.retain_grad()
                    preacts.append(activations)
            outputs.append(activations)
        return inputs, preacts, outputs


def ppo_loss(head, value, native_actions, old_logprobs, advantages, returns, old_values,
             log_scale, args):
    """The real clipped PPO objective, formed exactly as the trainer forms it."""
    alpha, beta = (F.softplus(head) + 1.0).chunk(2, dim=-1)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - log_scale).sum(-1)
    ratio = (newlogprob - old_logprobs).exp()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.max(-advantages * ratio,
                        -advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    newvalue = value.view(-1)
    clipped = old_values + (newvalue - old_values).clamp(-args.clip_coef, args.clip_coef)
    v_loss = 0.5 * torch.max((newvalue - returns).square(), (clipped - returns).square()).mean()
    return pg_loss + args.vf_coef * v_loss


def harvest(probe, agent, data, indices, args):
    """Layer inputs, preactivations and per-sample preactivation gradients."""
    obs, act, logp, adv, ret, val = (tensor[indices] for tensor in data)
    inputs, preacts, outputs = probe(obs)
    loss = ppo_loss(outputs[0], outputs[1], act, logp, adv, ret, val,
                    agent.log_action_scale, args)
    agent.zero_grad(set_to_none=True)
    loss.backward()
    return ([tensor.detach() for tensor in inputs],
            [tensor.detach() for tensor in preacts],
            [tensor.grad.detach() for tensor in preacts])


def row_gradients(inputs, signals):
    """(H, D) row gradients, i.e. sum_t d_th x_t for each unit."""
    return torch.einsum("th,td->hd", signals, inputs)


def cosine(left, right):
    return (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-30)


def state_bins(preacts, bins):
    """Equal-count bins of each unit's OWN preactivation. No tuned threshold."""
    samples = preacts.shape[0]
    order = preacts.argsort(0).argsort(0)
    return order.mul_(bins).div_(samples, rounding_mode="floor")


def gate_arms(inputs, signals, preacts, target_rows, bins, generator):
    """Closed-form optimal gates in [0,1]; the reduction is linear in the gate."""
    samples = preacts.shape[0]
    # <p_th, ghat_h>: whether sample t helps unit h, measured against the target
    usefulness = signals * (inputs @ target_rows.T)
    index = state_bins(preacts, bins)
    pooled = torch.zeros((bins, preacts.shape[1]), device=inputs.device)
    pooled.scatter_add_(0, index, usefulness)
    slope = torch.zeros_like(pooled)
    slope.scatter_add_(0, index, 1.0 - preacts.tanh().square())
    # Remove the component of the per-bin usefulness pattern that a unit's own
    # tanh derivative already explains; the rest is genuinely novel own-state
    # content. Both vectors are centred so only the SHAPE across bins is used.
    centred = pooled - pooled.mean(0, keepdim=True)
    direction = slope - slope.mean(0, keepdim=True)
    direction = direction / direction.norm(dim=0, keepdim=True).clamp_min(1e-30)
    orthogonal = centred - direction * (centred * direction).sum(0, keepdim=True)
    weights = (pooled > 0).to(inputs.dtype)
    arms = {
        "ungated": torch.ones_like(usefulness),
        "free": (usefulness > 0).to(inputs.dtype),
        "state": weights.gather(0, index),
        "state_orth": (orthogonal > 0).to(inputs.dtype).gather(0, index),
        "shared": (usefulness.sum(1, keepdim=True) > 0).to(inputs.dtype
                                                           ).expand_as(usefulness),
        "shuffled": weights.gather(
            0, torch.rand_like(weights).argsort(0)).gather(0, index),
    }
    return arms, samples


def decompose(inputs, signals, gate, ungated_rows):
    """Split the gated update into its LR term and its covariance term."""
    level = gate.mean(0)                                            # (H,)
    lr_term = level.unsqueeze(-1) * ungated_rows
    gated = row_gradients(inputs, signals * gate)
    covariance = gated - lr_term
    return gated, lr_term, covariance


def evaluate(inputs, signals, preacts, target_rows, score_rows, bins, generator):
    arms, samples = gate_arms(inputs, signals, preacts, target_rows, bins, generator)
    ungated_rows = row_gradients(inputs, signals)
    out = {}
    for name, gate in arms.items():
        gated, lr_term, covariance = decompose(inputs, signals, gate, ungated_rows)
        ratio = covariance.norm(dim=-1) / lr_term.norm(dim=-1).clamp_min(1e-30)
        out[name] = {
            "ratio_median": float(ratio.median()),
            "ratio_mean": float(ratio.mean()),
            "cos_in": float(cosine(gated.flatten(), target_rows.flatten())),
            "cos_held": float(cosine(gated.flatten(), score_rows.flatten())),
            "cos_lr_held": float(cosine(lr_term.flatten(), score_rows.flatten())),
            "gate_mean": float(gate.mean()),
            "gate_sd": float(gate.mean(0).std()),
        }
    return out


def shared_share(matrix, generator):
    """Variance fraction taken by the best shared per-sample rank-1 component.

    Columns are standardised first: otherwise the fit is captured by whichever
    unit has the largest gradient scale, which is a per-unit scale artifact and
    not shared structure. The permuted arm shuffles each column independently
    across samples, which is the matched null.
    """
    def fraction(values):
        values = values - values.mean(0, keepdim=True)
        values = values / values.std(0, keepdim=True).clamp_min(1e-30)
        energy = torch.linalg.svdvals(values.double()).square()
        return float(energy[0] / energy.sum().clamp_min(1e-30))

    return fraction(matrix), fraction(matrix.gather(0, torch.rand_like(matrix).argsort(0)))


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    probe_sizes = [int(value) for value in args.probe_sizes.split(",")]
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend="native",
                                  num_threads=min(args.env_threads, args.num_envs))
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    agent = Agent(envs).to(device)
    probe = Probe(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_norm = VectorObsNorm(args.num_envs, (obs_dim,))
    rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs = obs_norm.normalize(raw_obs)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    batch = args.num_envs * args.num_steps
    rollouts = max(args.total_steps // batch, args.stages)
    per_stage = max(rollouts // args.stages, 1)
    pool_rollouts = max((3 * max(probe_sizes) + batch - 1) // batch, 1)
    print(f"# {args.env_id}: {rollouts} rollouts x {batch} steps = {rollouts * batch} "
          f"env steps, train minibatch {args.minibatch_size}, probe sizes "
          f"{probe_sizes}, {args.bins} own-state bins, {args.replicates} replicates "
          f"x {args.stages} stages, lr {args.learning_rate}, seed {args.seed}")
    print(f"# tracked hidden layers: {', '.join(probe.names)}")
    print(f"# NOTE the benchmark recipe's optimizer step sees {args.minibatch_size} "
          f"samples, not {batch}; both regimes are in the batch-scaling table.")
    start = time.perf_counter()

    def collect():
        nonlocal next_obs
        steps, count = args.num_steps, args.num_envs
        with torch.no_grad():
            fields = {name: torch.zeros((steps, count) + shape, device=device)
                      for name, shape in (("obs", (obs_dim,)), ("act", (agent.action_dim,)),
                                          ("logp", ()), ("val", ()), ("rew", ()), ("done", ()))}
            for step in range(steps):
                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                alpha, beta, value = agent.get_policy_and_value(obs_t)
                native, physical = sample_beta_actions(alpha, beta, agent.action_low,
                                                       agent.action_high)
                fields["obs"][step] = obs_t
                fields["act"][step] = native
                fields["logp"][step] = agent.action_logprob(alpha, beta, native)
                fields["val"][step] = value.flatten()
                raw, reward, terms, truncs, infos = envs.step(
                    physical.cpu().numpy().reshape((count,) + agent.action_shape))
                fields["rew"][step] = torch.as_tensor(rew_norm.normalize(reward, terms),
                                                      dtype=torch.float32, device=device)
                next_obs, _ = obs_norm.normalize_step(raw, terms, truncs, infos)
                fields["done"][step] = torch.as_tensor(
                    np.maximum(terms, truncs).astype(np.float32), device=device)
            tail = agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32,
                                                   device=device)).flatten()
            advantages = torch.zeros_like(fields["rew"])
            running = torch.zeros_like(tail)
            for step in reversed(range(steps)):
                following = tail if step == steps - 1 else fields["val"][step + 1]
                nonterminal = 1.0 - fields["done"][step]
                delta = (fields["rew"][step] + args.gamma * nonterminal * following
                         - fields["val"][step])
                running = delta + args.gamma * args.gae_lambda * nonterminal * running
                advantages[step] = running
            returns = advantages + fields["val"]
        return (fields["obs"].flatten(0, 1), fields["act"].flatten(0, 1),
                fields["logp"].flatten(0, 1), advantages.flatten(), returns.flatten(),
                fields["val"].flatten())

    def train(data):
        for _ in range(args.update_epochs):
            order = torch.randperm(batch, device=device, generator=generator)
            for begin in range(0, batch, args.minibatch_size):
                indices = order[begin:begin + args.minibatch_size]
                obs, act, logp, adv, ret, val = (tensor[indices] for tensor in data)
                _, _, outputs = probe(obs)
                loss = ppo_loss(outputs[0], outputs[1], act, logp, adv, ret, val,
                                agent.log_action_scale, args)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

    def measure(pool, label):
        available = pool[0].shape[0]
        table = {}
        rank = {}
        for size in probe_sizes:
            if 3 * size > available:
                continue
            for replicate in range(args.replicates):
                order = torch.randperm(available, device=device, generator=generator)
                triple = [order[index * size:(index + 1) * size] for index in range(3)]
                harvested = [harvest(probe, agent, pool, indices, args) for indices in triple]
                for layer, name in enumerate(probe.names):
                    inputs, preacts = harvested[0][0][layer], harvested[0][1][layer]
                    signals = harvested[0][2][layer]
                    target = row_gradients(harvested[1][0][layer], harvested[1][2][layer])
                    score = row_gradients(harvested[2][0][layer], harvested[2][2][layer])
                    scores = evaluate(inputs, signals, preacts, target, score,
                                      args.bins, generator)
                    for arm, values in scores.items():
                        for key, value in values.items():
                            table.setdefault((size, name, arm), {}).setdefault(
                                key, []).append(value)
                    if size == max(probe_sizes) or 3 * size <= available < 3 * max(probe_sizes):
                        own = signals * (inputs @ row_gradients(inputs, signals).T)
                        observed, null = shared_share(own, generator)
                        rank.setdefault(name, {"obs": [], "null": []})
                        rank[name]["obs"].append(observed)
                        rank[name]["null"].append(null)
        report(label, table, rank)

    def report(label, table, rank):
        print(f"\n{label}")
        sizes = sorted({key[0] for key in table})
        print("  1+2+3. COVARIANCE RATIO ||cov term|| / ||LR term||, and held-out cosine")
        print(f"    {'B':>6} {'layer':<10} {'arm':<11} {'ratio med':>16} "
              f"{'cos held-out':>18} {'cos in-sample':>10} {'gate mean':>10} {'gate sd':>8}")
        for size in sizes:
            for name in probe.names:
                for arm in ARMS:
                    values = table.get((size, name, arm))
                    if values is None:
                        continue
                    print(f"    {size:>6} {name:<10} {arm:<11} "
                          f"{np.mean(values['ratio_median']):8.4f}+-"
                          f"{np.std(values['ratio_median']):.4f} "
                          f"{np.mean(values['cos_held']):9.5f}+-"
                          f"{np.std(values['cos_held']):.5f} "
                          f"{np.mean(values['cos_in']):10.5f} "
                          f"{np.mean(values['gate_mean']):10.4f} "
                          f"{np.mean(values['gate_sd']):8.4f}")
        print("  4. BATCH SCALING of the ratio, against the 1/sqrt(B) line that")
        print("     uncorrelated gate variance would follow (normalised at the smallest B)")
        for arm in ("free", "state", "state_orth", "shuffled"):
            row = []
            base = None
            for size in sizes:
                values = [np.mean(table[(size, name, arm)]["ratio_median"])
                          for name in probe.names if (size, name, arm) in table]
                if not values:
                    continue
                ratio = float(np.mean(values))
                base = base or (ratio, size)
                expected = base[0] * (base[1] / size) ** 0.5
                row.append(f"B={size}: {ratio:.4f} (1/sqrt(B) would give {expected:.4f})")
            print(f"    {arm:<11} " + "   ".join(row))
        if rank:
            print("  AUX. shared rank-1 share of the per-(unit,sample) usefulness matrix")
            for name, entry in rank.items():
                print(f"    {name:<10} shared {np.mean(entry['obs']):.4f}+-"
                      f"{np.std(entry['obs']):.4f}   permuted null "
                      f"{np.mean(entry['null']):.4f}   residual per-unit "
                      f"{1.0 - np.mean(entry['obs']):.4f}")

    pool_data, returns_seen = [], []
    for rollout in range(rollouts):
        data = collect()
        returns_seen.append(float(data[4].mean()))
        pool_data.append(data)
        if len(pool_data) > pool_rollouts:
            pool_data.pop(0)
        train(data)
        if (rollout + 1) % per_stage == 0 or rollout == rollouts - 1:
            if len(pool_data) >= pool_rollouts:
                pool = tuple(torch.cat([entry[field] for entry in pool_data], dim=0)
                             for field in range(6))
                measure(pool, f"=== steps {(rollout + 1) * batch / 1e6:.2f}M  "
                              f"(mean return proxy "
                              f"{np.mean(returns_seen[-per_stage:]):.1f}) ===")
    envs.close()
    print(f"\n# wall time {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()

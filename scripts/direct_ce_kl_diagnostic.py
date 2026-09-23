"""Observe frozen v20 training, or replay captured iterations at their saved LR.

Training: python scripts/direct_ce_kl_diagnostic.py <the unchanged v20 arguments>
Replay:   python scripts/direct_ce_kl_diagnostic.py replay SNAPSHOT.pt
All model execution requires CUDA and compilation; run through mlq.
Diagnostic settings use DIRECT_CE_DIAG_PREFIX, DIRECT_CE_DIAG_CAPTURE_STEPS,
DIRECT_CE_DIAG_SPIKES (default 3), and DIRECT_CE_DIAG_THRESHOLD (default 0.1).
These affect recording only. No trainer source, loss, RNG, or update is replaced.
"""

import copy
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TRAINER_MODULE = "cleanrl.ppo_continuous_action_residual_stiglu_ngpt_readout_geometry_v20"
BATCH_LOCALS = {
    "observations": "b_obs", "native_actions": "b_native",
    "old_logprobs": "b_logprobs", "advantages": "b_advantages",
    "target_probs": "b_target_probs", "old_alpha": "b_alpha", "old_beta": "b_beta",
}
QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)
VARIANTS = (
    "baseline", "zero_actor_first_moment", "tangent_actor_update",
    "no_actor_projection", "without_advantage_tail",
    "without_negative_high_ratio", "without_score_weight_tail",
    "positive_advantages_only", "negative_advantages_only",
)


def owned(value, cpu=False):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True) if cpu else value.detach().clone()
    if isinstance(value, dict):
        return {key: owned(item, cpu) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(owned(item, cpu) for item in value)
    return copy.deepcopy(value)


def beta_kl(a, b, c, d):
    """Analytic KL(Beta(a,b)||Beta(c,d)); FP64 reduces near-zero cancellation."""
    a, b, c, d = (x.double() for x in (a, b, c, d))
    old_sum, new_sum = a + b, c + d
    return (c.lgamma() + d.lgamma() + old_sum.lgamma()
            - a.lgamma() - b.lgamma() - new_sum.lgamma()
            + (a - c) * a.digamma() + (b - d) * b.digamma()
            + (new_sum - old_sum) * old_sum.digamma()).sum(-1)


def correlation(x, y):
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    return (x * y).mean() / (x.square().mean() * y.square().mean()).sqrt().clamp_min(1e-30)


def sorted_quantile(ordered, q):
    index = (ordered.numel() - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def quantiles(result, prefix, values):
    ordered = values.sort().values
    for label in QUANTILES:
        result[f"{prefix}/q{label:g}"] = sorted_quantile(ordered, label)


def masked_sum(values, mask):
    return torch.where(mask, values, 0).sum()


def make_diagnostics(agent, args):
    @torch.no_grad()
    def state(observations, native_actions, old_logprobs, advantages, old_alpha, old_beta):
        # Explicit residual stages reuse the original actor modules; no hooks on
        # live compiled modules and no sampling or optimizer-state mutation.
        actor = agent.actor
        first = actor.first(observations)
        h = F.normalize(first, p=2, dim=-1)
        second = actor.second(8.0 * h)
        branch = F.normalize(second, p=2, dim=-1)
        residual = h + branch
        logits = actor.readout_gain * actor.readout_scale * actor.head(
            8.0 * F.normalize(residual, p=2, dim=-1))
        alpha, beta = (F.softplus(logits) + 1.0).chunk(2, dim=-1)
        kl = beta_kl(old_alpha, old_beta, alpha, beta)
        old_c, new_c = old_alpha + old_beta, alpha + beta
        old_m, new_m = old_alpha / old_c, alpha / new_c
        # These are distribution-level counterfactuals, not an additive KL split.
        mean_only = beta_kl(old_alpha, old_beta, new_m * old_c, (1 - new_m) * old_c)
        concentration_only = beta_kl(old_alpha, old_beta, old_m * new_c, (1 - old_m) * new_c)
        logprob = agent.action_logprob(alpha, beta, native_actions)
        logratio = logprob - old_logprobs
        ratio = logratio.exp()
        adv = advantages
        if args.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        active = ((adv > 0) & (ratio <= 1 + args.clip_coef_upper)) | (
            (adv < 0) & (ratio >= 1 - args.clip_coef))
        abs_adv = adv.abs()
        top = kl >= sorted_quantile(kl.sort().values, 0.99)
        adv_tail = abs_adv >= sorted_quantile(abs_adv.sort().values, 0.99)
        # Scalar logprob derivative magnitude; not a parameter-gradient norm.
        score_weight = abs_adv * ratio * active
        score_tail = score_weight >= sorted_quantile(score_weight.sort().values, 0.99)
        negative_high = (adv < 0) & (ratio > 1 + args.clip_coef_upper)
        weight_sum = score_weight.sum().clamp_min(1e-30)
        result = {
            "kl/mean": kl.mean(), "kl/mean_only": mean_only.mean(),
            "kl/concentration_only": concentration_only.mean(),
            "kl/top1pct_mass_fraction": masked_sum(kl, top) / kl.sum().clamp_min(1e-30),
            "kl/adv_tail_mass_fraction": masked_sum(kl, adv_tail) / kl.sum().clamp_min(1e-30),
            "policy/mean_rms_displacement": (new_m - old_m).square().mean().sqrt(),
            "policy/log_concentration_rms_displacement": (new_c.log() - old_c.log()).square().mean().sqrt(),
            "ppo/active_fraction": active.float().mean(),
            "ppo/active_positive_fraction": (active & (adv > 0)).float().mean(),
            "ppo/active_negative_fraction": (active & (adv < 0)).float().mean(),
            "ppo/score_weight_tail_fraction": masked_sum(score_weight, adv_tail) / weight_sum,
            "ppo/score_weight_top1pct_fraction": masked_sum(score_weight, score_tail) / weight_sum,
            "ppo/negative_high_ratio_weight_fraction": masked_sum(score_weight, negative_high) / weight_sum,
            "ppo/negative_high_ratio_fraction": negative_high.float().mean(),
            "ppo/negative_high_ratio_weight_sum": masked_sum(score_weight, negative_high),
            "ppo/score_weight_sum": score_weight.sum(),
            "ppo/score_weight_max": score_weight.max(),
            "ppo/logratio_max": logratio.max(),
            "ppo/logratio_min": logratio.min(),
            "ppo/ratio_max": ratio.max(),
            "advantage/abs_mean": abs_adv.mean(),
            "advantage/tail_abs_mass_fraction": masked_sum(abs_adv, adv_tail) / abs_adv.sum().clamp_min(1e-30),
            "advantage/tail_square_mass_fraction": masked_sum(adv.square(), adv_tail) / adv.square().sum().clamp_min(1e-30),
            "integrity/alpha_max_error": (alpha - old_alpha).abs().max(),
            "integrity/beta_max_error": (beta - old_beta).abs().max(),
            "integrity/logprob_max_error": (logprob - old_logprobs).abs().max(),
        }
        quantiles(result, "kl", kl)
        quantiles(result, "advantage/abs", abs_adv)
        quantiles(result, "ppo/ratio", ratio)
        quantiles(result, "ppo/logratio", logratio)
        for name, values in (("first", first), ("second", second), ("residual", residual)):
            norms = values.norm(dim=-1)
            quantiles(result, f"normalization/{name}", norms)
            result[f"normalization/{name}/top_kl_mean"] = masked_sum(norms, top) / top.sum().clamp_min(1)
            result[f"normalization/{name}/inverse_norm_kl_correlation"] = correlation(norms.clamp_min(1e-12).reciprocal(), kl)
        return result

    # Fixed-shape sorted quantiles and masked sums remain fullgraph-compatible.
    return torch.compile(state, fullgraph=True, options={"triton.cudagraphs": False})


def actor_matrices(agent):
    result = {}
    for stage_name in ("first", "second"):
        for layer, dim in (("gate", 1), ("up", 1), ("down", 0)):
            result[f"{stage_name}.0.{layer}.weight"] = dim
    result["head.0.weight"] = 1
    return result


def vector_stats(parameters, optimizer):
    grad2 = torch.zeros((), device="cuda")
    moment2, dot = grad2.clone(), grad2.clone()
    opposing = grad2.clone()
    count = 0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        m = optimizer.state.get(p, {}).get("exp_avg")
        grad2 += g.square().sum()
        if m is not None:
            moment2 += m.square().sum()
            dot += (m * g).sum()
            opposing += ((m * g) < 0).sum()
        count += p.numel()
    return {
        "actor/gradient_norm": grad2.sqrt(), "adam/first_moment_norm": moment2.sqrt(),
        "adam/first_moment_gradient_cosine": dot / (grad2 * moment2).sqrt().clamp_min(1e-30),
        "adam/opposing_coordinate_fraction": opposing / max(count, 1),
    }


def update_stats(agent, before, gradients, prefix):
    total2 = torch.zeros((), device="cuda")
    grad2, dot, parameter2, radial2 = (total2.clone() for _ in range(4))
    axes = actor_matrices(agent)
    result = {}
    for name, parameter in agent.actor.named_parameters():
        change = parameter.detach() - before[name]
        total2 += change.square().sum()
        parameter2 += before[name].square().sum()
        grad2 += gradients[name].square().sum()
        dot += (change * gradients[name]).sum()
        if name in axes:
            dim = axes[name]
            old = before[name]
            radial = old * (change * old).sum(dim=dim, keepdim=True) / old.square().sum(dim=dim, keepdim=True).clamp_min(1e-30)
            radial2 += radial.square().sum()
        result[f"{prefix}/parameter/{name}/delta_norm"] = change.norm()
    result.update({f"{prefix}/norm": total2.sqrt(),
                   f"{prefix}/relative_norm": (total2 / parameter2.clamp_min(1e-30)).sqrt(),
                   f"{prefix}/gradient_cosine": dot / (total2 * grad2).sqrt().clamp_min(1e-30),
                   f"{prefix}/radial_fraction": (radial2 / total2.clamp_min(1e-30)).sqrt()})
    return result


def floats(record):
    keys = list(record)
    values = torch.stack([record[key].detach().double() for key in keys]).cpu().tolist()
    return dict(zip(keys, values))


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class DiagnosticWriter:
    def __init__(self):
        self.prefix = None
        self.summary = None
        self.records = None
        self.handles = []
        self.pending = None
        self.max_kl = -float("inf")
        self.spikes = 0
        self.low_saved = False
        self.steps = {int(x) for x in os.environ.get("DIRECT_CE_DIAG_CAPTURE_STEPS", "13745792,22068864,22085248").split(",") if x}
        self.spike_limit = int(os.environ.get("DIRECT_CE_DIAG_SPIKES", "3"))
        self.threshold = float(os.environ.get("DIRECT_CE_DIAG_THRESHOLD", "0.1"))
        self.snapshots = {}

    def start(self, local):
        args = local["args"]
        if not args.compile or not args.cuda or args.variance_gain:
            raise ValueError("Diagnostic requires compiled CUDA direct CE (--no-variance-gain)")
        if self.prefix is None:
            self.prefix = Path(os.environ.get("DIRECT_CE_DIAG_PREFIX", f"runs/direct_ce_kl_diagnostic_{local['run_name']}"))
            self.prefix.parent.mkdir(parents=True, exist_ok=True)
            self.records = self.prefix.with_suffix(".jsonl").open("w")
            self.summary = self.prefix.with_suffix(".json")
            self.writer = local["writer"]
            self.evaluate = make_diagnostics(local["agent"], args)
            self.handles = [local["optimizer"].register_step_pre_hook(self.before_step),
                            local["optimizer"].register_step_post_hook(self.after_step)]
            print(f"DIRECT_CE_DIAGNOSTICS json={self.summary} updates={self.records.name} tensorboard={local['run_name']} snapshots={self.prefix}_*.pt", flush=True)
        self.local = local
        self.pending = {"format_version": 1, "trainer_module": TRAINER_MODULE,
                        "args": owned(vars(args)), "step": int(local["global_step"]),
                        "iteration": int(local["iteration"]),
                        "model_state": owned(local["agent"].state_dict()),
                        "optimizer_state": owned(local["optimizer"].state_dict()),
                        "minibatch_indices": [], "epochs": [],
                        **{key: owned(local[name]) for key, name in BATCH_LOCALS.items()}}
        self.update_records = []
        self.initial = floats(self.state())
        # The host mirror intentionally remains the trainer's existing CPU actor;
        # diagnostic model execution itself is CUDA, never a CPU fallback.
        count = args.num_envs
        obs = self.pending["observations"][:count]
        host = local["host_actor"](obs.cpu().numpy()).copy()
        with torch.no_grad():
            gpu = self.actor_logits(obs)
            self.initial["integrity/host_logits_max_error"] = float((gpu - torch.as_tensor(host, device="cuda")).abs().max())
        self.initial["integrity/old_alpha_min"] = float(self.pending["old_alpha"].min())
        self.initial["integrity/old_beta_min"] = float(self.pending["old_beta"].min())
        self.initial["integrity/native_actions_min"] = float(self.pending["native_actions"].min())
        self.initial["integrity/native_actions_max"] = float(self.pending["native_actions"].max())

    def actor_logits(self, obs):
        if not hasattr(self, "compiled_actor"):
            self.compiled_actor = torch.compile(self.local["agent"].actor, fullgraph=True, options={"triton.cudagraphs": False})
        return self.compiled_actor(obs)

    def state(self):
        batch = self.pending
        return self.evaluate(*(batch[name] for name in ("observations", "native_actions", "old_logprobs", "advantages", "old_alpha", "old_beta")))

    @torch.no_grad()
    def before_step(self, optimizer, args, kwargs):
        agent = self.local["agent"]
        self.before = {name: p.detach().clone() for name, p in agent.actor.named_parameters()}
        self.gradients = {name: p.grad.detach().clone() for name, p in agent.actor.named_parameters()}
        self.current = {f"pre/{key}": value for key, value in self.state().items()}
        self.current.update(vector_stats(tuple(agent.actor.parameters()), optimizer))

    @torch.no_grad()
    def after_step(self, optimizer, args, kwargs):
        agent = self.local["agent"]
        self.current.update({f"adam/{key}": value for key, value in self.state().items()})
        self.current.update(update_stats(agent, self.before, self.gradients, "adam_delta"))
        self.unprojected = {name: p.detach().clone() for name, p in agent.actor.named_parameters()}

    @torch.no_grad()
    def after_projection(self):
        agent = self.local["agent"]
        self.current.update({f"projected/{key}": value for key, value in self.state().items()})
        self.current.update(update_stats(agent, self.before, self.gradients, "actual_delta"))
        self.current.update(update_stats(agent, self.unprojected, self.gradients, "projection_delta"))
        self.update_records.append(self.current)

    def finish(self, local):
        step = self.pending["step"]
        records = [floats(record) for record in self.update_records]
        immutable = {key: bool(torch.equal(self.pending[key], local[name])) for key, name in BATCH_LOCALS.items()}
        peak = max(record["projected/kl/mean"] for record in records)
        record = {"step": step, "iteration": self.pending["iteration"], "initial": self.initial,
                  "immutable_batch": immutable, "updates": records, "peak_projected_kl": peak}
        self.records.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        self.records.flush()
        for name, value in self.initial.items():
            self.writer.add_scalar(f"direct_ce_diagnostic/initial/{name}", value, step)
        # Per-update axes retain epoch-level evidence; iteration metrics use env steps.
        count = len(records)
        for update, metrics in enumerate(records):
            axis = (self.pending["iteration"] - 1) * count + update
            for name, value in metrics.items():
                if "/parameter/" not in name:
                    self.writer.add_scalar(f"direct_ce_updates/{name}", value, axis)
        self.writer.add_scalar("direct_ce_diagnostic/peak_projected_kl", peak, step)
        self.writer.add_scalar("direct_ce_diagnostic/batch_immutable", int(all(immutable.values())), step)
        reasons = []
        if not self.low_saved and peak < self.threshold:
            reasons.append("low")
            self.low_saved = True
        if peak > self.max_kl:
            reasons.append("top")
            self.max_kl = peak
        if peak > self.threshold and self.spikes < self.spike_limit:
            reasons.append(f"spike{self.spikes + 1}")
            self.spikes += 1
        if step in self.steps:
            reasons.append(f"step{step}")
        if not all(immutable.values()):
            reasons.append(f"integrity_step{step}")
        if reasons:
            self.pending["evidence"] = record
            self.pending["post_model_state"] = owned(local["agent"].state_dict())
            self.pending["post_optimizer_state"] = owned(local["optimizer"].state_dict())
            snapshot = owned(self.pending, cpu=True)
            for reason in reasons:
                path = Path(f"{self.prefix}_{reason}.pt")
                torch.save(snapshot, path)
                self.snapshots[reason] = {"path": str(path), "step": step, "peak_projected_kl": peak}
                print(f"DIRECT_CE_SNAPSHOT reason={reason} step={step} peak_kl={peak:.8g} path={path}", flush=True)
        self.summary.write_text(json.dumps(json_safe({"trainer": TRAINER_MODULE, "args": self.pending["args"],
            "last_step": step, "max_projected_kl": self.max_kl, "snapshots": self.snapshots,
            "requested_steps": sorted(self.steps), "updates_jsonl": self.records.name,
            "tensorboard": self.writer.log_dir,
            "note": "Recording only. Matched mean/concentration KLs are not an additive decomposition."}), indent=2, allow_nan=False) + "\n")
        self.pending = None
        self.local = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.records is not None:
            self.records.close()


def train():
    trainer = importlib.import_module(TRAINER_MODULE)
    original = trainer.device_minibatches
    original_validate = trainer.validate_args
    diagnostic = DiagnosticWriter()

    def validate(args):
        args = original_validate(args)
        if not args.compile or args.variance_gain:
            raise ValueError("Diagnostic requires compiled CUDA direct CE (--no-variance-gain)")
        return args

    def minibatches(*args, **kwargs):
        frame = inspect.currentframe()
        caller = frame.f_back
        del frame
        if caller is None or caller.f_code is not trainer.main.__code__:
            raise RuntimeError("Expected device_minibatches directly inside frozen v20.main")
        try:
            local = caller.f_locals
            if local["updates"] == 0:
                diagnostic.start(local)
            for indices in original(*args, **kwargs):
                diagnostic.pending["minibatch_indices"].append(indices.detach().clone())
                diagnostic.pending["epochs"].append(int(local["epoch"]))
                yield indices
                diagnostic.after_projection()
            local = caller.f_locals
            if local["updates"] == local["max_updates"]:
                diagnostic.finish(local)
        finally:
            del caller

    trainer.device_minibatches = minibatches
    trainer.validate_args = validate
    try:
        trainer.main()
    finally:
        trainer.device_minibatches = original
        trainer.validate_args = original_validate
        diagnostic.close()


def replay(path):
    trainer = importlib.import_module(TRAINER_MODULE)
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    args = trainer.validate_args(trainer.Args(**snapshot["args"]))
    if not torch.cuda.is_available() or not args.compile or args.variance_gain:
        raise RuntimeError("Replay requires CUDA, compilation, and direct CE snapshot")
    trainer.configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
    state = snapshot["model_state"]
    obs_dim = snapshot["observations"].shape[-1]
    envs = SimpleNamespace(single_observation_space=trainer.gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
                           single_action_space=trainer.gym.spaces.Box(state["action_low"].numpy(), state["action_high"].numpy(), dtype=np.float32))
    agent = trainer.Agent(envs, args).cuda()
    optimizer = torch.optim.Adam(tuple(agent.parameters()), lr=args.learning_rate, eps=1e-5, fused=True)
    batch = {key: snapshot[key].cuda() for key in BATCH_LOCALS}
    evaluate = make_diagnostics(agent, args)
    def state_metrics():
        return evaluate(*(batch[key] for key in ("observations", "native_actions", "old_logprobs", "advantages", "old_alpha", "old_beta")))
    def loss_model(obs, native, logprob, adv, targets, alpha, beta):
        return trainer.ppo_loss(agent, obs, native, logprob, adv, targets, args, alpha, beta)
    loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
    attribution_args = copy.copy(args)
    attribution_args.ent_coef = 0.0
    attribution_args.vf_coef = 0.0
    def attribution_loss(obs, native, logprob, adv, targets, alpha, beta):
        return trainer.ppo_loss(agent, obs, native, logprob, adv, targets, attribution_args, alpha, beta)[0]
    attribution_loss = torch.compile(attribution_loss, fullgraph=True, dynamic=False,
                                     options={"triton.cudagraphs": False})
    @torch.no_grad()
    def offending_samples(obs, native, logprob, adv):
        alpha, beta = (F.softplus(agent.actor(obs)) + 1).chunk(2, dim=-1)
        ratio = (agent.action_logprob(alpha, beta, native) - logprob).exp()
        active = ((adv > 0) & (ratio <= 1 + args.clip_coef_upper)) | ((adv < 0) & (ratio >= 1 - args.clip_coef))
        weight = adv.abs() * ratio * active
        return ((adv < 0) & (ratio > 1 + args.clip_coef_upper),
                weight >= sorted_quantile(weight.sort().values, 0.99))
    offending_samples = torch.compile(offending_samples, fullgraph=True, options={"triton.cudagraphs": False})
    normalize = torch.compile(agent.normalize_matrices, fullgraph=True, options={"triton.cudagraphs": False})
    axes = actor_matrices(agent)
    indices_list = [indices.cuda() for indices in snapshot["minibatch_indices"]]
    prefix = Path(os.environ.get("DIRECT_CE_DIAG_PREFIX", str(Path(path).with_suffix("")) + "_replay"))
    writer = SummaryWriter(str(prefix) + "_tb")
    output = {"snapshot": str(path), "step": snapshot["step"], "variants": {},
              "note": "Fixed-batch diagnostic counterfactuals at identical saved LR, not training results. No LR arms.",
              "saved_learning_rate": snapshot["optimizer_state"]["param_groups"][0]["lr"]}
    # Offending sets come from the matched baseline at each update, not a moving
    # threshold in the counterfactual. This isolates those samples' contributions.
    baseline_masks = []
    try:
        for variant in VARIANTS:
            agent.load_state_dict(state)
            optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer_state"]))
            optimizer.zero_grad(set_to_none=True)
            if variant == "zero_actor_first_moment":
                for p in agent.actor.parameters():
                    if "exp_avg" in optimizer.state[p]:
                        optimizer.state[p]["exp_avg"].zero_()
            advantages = batch["advantages"].clone()
            if variant == "without_advantage_tail":
                tail = advantages.abs() >= torch.quantile(advantages.abs(), 0.99)
                advantages[tail] = 0
            elif variant == "positive_advantages_only":
                advantages = advantages.clamp_min(0)
            elif variant == "negative_advantages_only":
                advantages = advantages.clamp_max(0)
            if args.norm_adv and variant in VARIANTS[4:]:
                raise ValueError("Advantage attribution arms require the v20 directCE no-advantage-normalization setup")
            rows = []
            for update, indices in enumerate(indices_list):
                torch.compiler.cudagraph_mark_step_begin()
                pre = {name: p.detach().clone() for name, p in agent.actor.named_parameters()}
                pre_metrics = floats(state_metrics())
                if variant == "baseline":
                    baseline_masks.append(tuple(mask.clone() for mask in offending_samples(
                        batch["observations"], batch["native_actions"], batch["old_logprobs"], batch["advantages"])))
                if variant in ("without_negative_high_ratio", "without_score_weight_tail"):
                    mask = baseline_masks[update][0 if variant == "without_negative_high_ratio" else 1]
                    advantages = batch["advantages"].masked_fill(mask, 0)
                loss, metrics = loss_model(batch["observations"][indices], batch["native_actions"][indices],
                    batch["old_logprobs"][indices], advantages[indices], batch["target_probs"][indices],
                    batch["old_alpha"][indices], batch["old_beta"][indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                trainer.apply_gradient_clipping(tuple(agent.parameters()), (agent.critic.readout_gain,), args.max_grad_norm, args.grad_clip)
                gradients = {name: p.grad.detach().clone() for name, p in agent.actor.named_parameters()}
                grad_metrics = floats(vector_stats(tuple(agent.actor.parameters()), optimizer))
                attribution = {}
                if variant == "baseline" and not args.norm_adv:
                    # Independent compiled autograd evaluations return tensors;
                    # they never accumulate into or replace live optimizer grads.
                    masks = {
                        "positive": batch["advantages"] > 0,
                        "negative": batch["advantages"] < 0,
                        "raw_advantage_tail": batch["advantages"].abs() >= torch.quantile(batch["advantages"].abs(), 0.99),
                        "negative_high_ratio": baseline_masks[update][0],
                        "score_weight_tail": baseline_masks[update][1],
                    }
                    full = torch.cat([g.flatten() for g in gradients.values()])
                    for label, mask in masks.items():
                        masked_adv = batch["advantages"] * mask
                        component_loss = attribution_loss(
                            batch["observations"][indices], batch["native_actions"][indices],
                            batch["old_logprobs"][indices], masked_adv[indices],
                            batch["target_probs"][indices], batch["old_alpha"][indices], batch["old_beta"][indices])
                        component = torch.cat([g.flatten() for g in torch.autograd.grad(
                            component_loss, tuple(agent.actor.parameters()))])
                        dot = (component * full).sum()
                        attribution[label] = floats({
                            "norm": component.norm(),
                            "cosine_with_actual_gradient": dot / (component.norm() * full.norm()).clamp_min(1e-30),
                            "signed_projection_fraction": dot / full.square().sum().clamp_min(1e-30),
                            "sample_fraction": mask[indices].float().mean(),
                        })
                optimizer.step()
                raw_metrics = floats(state_metrics())
                with torch.no_grad():
                    if variant == "tangent_actor_update":
                        for name, p in agent.actor.named_parameters():
                            if name in axes:
                                dim, old = axes[name], pre[name]
                                delta = p - old
                                p.sub_(old * (delta * old).sum(dim=dim, keepdim=True) / old.square().sum(dim=dim, keepdim=True).clamp_min(1e-30))
                    unprojected = {name: p.detach().clone() for name, p in agent.actor.named_parameters()}
                    normalize()
                    if variant == "no_actor_projection":
                        for name, p in agent.actor.named_parameters():
                            p.copy_(unprojected[name])
                post_metrics = floats(state_metrics())
                row = {"update": update, "epoch": snapshot["epochs"][update], "pre": pre_metrics,
                       "raw_adam": raw_metrics, "post": post_metrics, "gradient": grad_metrics,
                       "update_geometry": floats(update_stats(agent, pre, gradients, "actual_delta")),
                       "same_state_policy_gradient_attribution": attribution}
                rows.append(row)
                for name, value in post_metrics.items():
                    writer.add_scalar(f"{variant}/{name}", value, update)
            baseline_error = None
            if variant == "baseline":
                baseline_error = max(float((value - snapshot["post_model_state"][key].to(value.device)).abs().max())
                                     for key, value in agent.state_dict().items() if value.is_floating_point())
            output["variants"][variant] = {"updates": rows, "baseline_post_model_max_abs_error": baseline_error}
            prefix.with_suffix(".json").write_text(json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n")
            print(f"DIRECT_CE_REPLAY variant={variant} final_kl={rows[-1]['post']['kl/mean']:.8g} baseline_error={baseline_error}", flush=True)
    finally:
        writer.close()
    print(f"DIRECT_CE_REPLAY_OUTPUT json={prefix.with_suffix('.json')} tensorboard={prefix}_tb", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "replay":
        if len(sys.argv) != 3:
            raise SystemExit("usage: direct_ce_kl_diagnostic.py replay SNAPSHOT.pt")
        replay(sys.argv[2])
    else:
        train()

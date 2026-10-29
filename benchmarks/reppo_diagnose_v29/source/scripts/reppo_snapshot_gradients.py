#!/usr/bin/env python3
"""Decompose v25/v27 critic gradients on an independent saved oracle rollout.

Queue with mlq. This observes a frozen checkpoint; it never trains or resumes it.
The saved expected-soft targets are critic-bootstrapped, not simulator oracle labels.
"""
import argparse
import glob
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reppo_latentroll_oracle import load_trainer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.rollout_graph import graph_compile


def group_gradients(critic, losses, coefficients):
    """Report every objective's raw/weighted group norms and combined-gradient shares."""
    named = list(critic.named_parameters())
    group_names = sorted({name.split('.')[0] for name, _ in named})
    gradients = {}
    for loss_name, loss in losses.items():
        values = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        gradients[loss_name] = {
            group: torch.cat([(torch.zeros_like(p) if grad is None else grad).flatten()
                              for (name, p), grad in zip(named, values) if name.split('.')[0] == group])
            for group in group_names}
    metrics = {f'loss/{name}': value.detach() for name, value in losses.items()}
    totals = {group: sum(coefficients[name] * grad[group] for name, grad in gradients.items())
              for group in group_names}
    total_squared = sum(value.square().sum() for value in totals.values())
    metrics['total/grad_norm'] = total_squared.sqrt()
    for group in group_names:
        metrics[f'group/{group}/total_norm'] = totals[group].norm()
        metrics[f'group/{group}/squared_norm_share'] = totals[group].square().sum() / total_squared.clamp_min(1e-20)
        for name, grads in gradients.items():
            metrics[f'group/{group}/{name}_raw_norm'] = grads[group].norm()
            metrics[f'group/{group}/{name}_weighted_norm'] = grads[group].norm() * abs(coefficients[name])
        for left, right in itertools.combinations(gradients, 2):
            # Zero-gradient objectives have cosine 0 by convention (not evidence of orthogonality).
            metrics[f'group/{group}/{left}_{right}_cosine'] = F.cosine_similarity(
                gradients[left][group], gradients[right][group], dim=0, eps=1e-12)
    for left, right in itertools.combinations(gradients, 2):
        dot = sum((gradients[left][g] * gradients[right][g]).sum() for g in group_names)
        left_norm = sum(gradients[left][g].square().sum() for g in group_names).sqrt()
        right_norm = sum(gradients[right][g].square().sum() for g in group_names).sqrt()
        metrics[f'total/{left}_{right}_cosine'] = dot / (left_norm * right_norm).clamp_min(1e-20)
    return {name: float(value.detach()) for name, value in metrics.items()}


def run(cli):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit this diagnostic through mlq')
    if min(cli.rows, cli.batches) < 1:
        raise ValueError('rows and batches must be positive')
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(cli.seed)
    snapshot = torch.load(cli.snapshot, map_location='cpu', weights_only=False)
    dataset = torch.load(cli.dataset, map_location='cpu', weights_only=False)
    chunk = dataset['chunk']
    if chunk.get('target_formula') != 'expected-soft-lambda-v25-v27':
        raise ValueError('Requires a dataset with corrected v25/v27 expected-soft targets')
    if Path(dataset['meta']['snapshot']).resolve() != cli.snapshot.resolve():
        raise ValueError('Snapshot differs from dataset provenance')
    if int(dataset['meta']['global_step']) != int(snapshot['global_step']):
        raise ValueError('Snapshot step differs from dataset provenance')
    trainer, trainer_file = load_trainer(snapshot, cli.trainer)
    args = SimpleNamespace(**snapshot['args'])
    device = torch.device('cuda')
    obs_dim, act_dim = chunk['raw_obs'].shape[-1], chunk['actions'].shape[-1]
    critic = trainer.Critic(obs_dim, act_dim, args, device).to(device)
    critic.load_state_dict(snapshot['critic'])
    actor = trainer.Actor(obs_dim, act_dim, args).to(device)
    actor.net.load_state_dict(snapshot['actor_net'])
    mean = torch.as_tensor(snapshot['obs_means'], device=device)
    scale = (torch.as_tensor(snapshot['obs_variances'], device=device) + snapshot['obs_epsilon']).sqrt()
    env_rows = torch.as_tensor(chunk['row'], device=device, dtype=torch.long)
    def normalize(raw):
        return ((raw.to(device) - mean[env_rows]) / scale[env_rows]).clamp(-args.obs_clip, args.obs_clip).float()
    obs, next_obs = normalize(chunk['raw_obs']), normalize(chunk['raw_next_obs'])
    actions = chunk['actions'].to(device)
    targets, rewards = chunk['targets'].to(device), chunk['rewards'].to(device)
    count, n, steps = len(obs), int(chunk['num_envs']), int(chunk['num_steps'])
    if cli.rows * cli.batches > count:
        raise ValueError('Requested disjoint minibatches exceed saved rollout rows')
    termination = chunk.get('terminations', torch.zeros(count)).to(device)
    cuts = torch.maximum(chunk['truncations'], chunk['ends']).to(device)
    if hasattr(critic, 'dynamics'):
        horizon = args.aux_horizon
        survives = ((1 - termination) * (1 - cuts)).view(steps, n)
        chains, chain = [], torch.ones_like(survives)
        for k in range(horizon):
            chain = chain * torch.cat((survives[k:], torch.zeros_like(survives[:k])))
            chains.append(chain.flatten())
        chains = torch.stack(chains)
    else:
        with torch.no_grad():
            sampled_next = graph_compile(lambda value: trainer.squashed_sample(*actor(value))[0])(next_obs)
            next_features = graph_compile(critic.features)(next_obs, sampled_next)
    permutation = torch.randperm(count, device=device, generator=torch.Generator(device=device).manual_seed(cli.seed))
    coefficients = {'ce': 1., 'latent': args.aux_loss_coef}
    if hasattr(critic, 'dynamics'):
        coefficients['reward'] = args.reward_loss_coef
    def loss_terms(rows):
        features = critic.features(obs[rows], actions[rows])
        logits = critic.logits(features)
        feature_mean = features.detach().mean(0)
        extra = {'feature/rms': features.detach().square().mean().sqrt(),
                 'feature/mean_vector_rms': feature_mean.square().mean().sqrt(),
                 'feature/centered_rms': (features.detach() - feature_mean).square().mean().sqrt()}
        losses = {'ce': -(critic.hl.project(targets[rows]) * logits.log_softmax(-1)).sum(-1).mean()}
        if hasattr(critic, 'dynamics'):
            later = (rows[None] + n * torch.arange(1, horizon + 1, device=device)[:, None]).clamp_max(count - 1)
            mask = chains[:, rows]
            with torch.no_grad():
                desired = critic.features(obs[later], actions[later])
            latents = [features]
            for k in range(horizon):
                latents.append(critic.transition(latents[-1], actions[later[k]]))
            latents = torch.stack(latents)
            errors = (latents[1:] - desired).square().mean(-1)
            losses['latent'] = (mask * errors).sum() / mask.sum().clamp_min(1.)
            for k in range(horizon):
                valid = mask[k].sum().clamp_min(1.)
                target_mean = (desired[k] * mask[k, :, None]).sum(0) / valid
                target_energy = (desired[k].square().mean(-1) * mask[k]).sum() / valid
                target_variance = ((desired[k] - target_mean).square().mean(-1) * mask[k]).sum() / valid
                mse = (errors[k].detach() * mask[k]).sum() / valid
                prefix = f'latent/h{k + 1}'
                extra[f'{prefix}/valid_rows'] = mask[k].sum()
                extra[f'{prefix}/mse'] = mse
                extra[f'{prefix}/target_rms'] = target_energy.sqrt()
                extra[f'{prefix}/target_mean_vector_rms'] = target_mean.square().mean().sqrt()
                extra[f'{prefix}/target_centered_rms'] = target_variance.sqrt()
                extra[f'{prefix}/energy_normalized_mse'] = mse / target_energy.clamp_min(1e-12)
                extra[f'{prefix}/variance_normalized_mse'] = mse / target_variance.clamp_min(1e-12)
            reward_rows = torch.cat((rows[None], later[:-1]))
            reward_mask = torch.cat((torch.ones_like(mask[:1]), mask[:-1]))
            losses['reward'] = (reward_mask * (critic.reward(latents[:-1]) - rewards[reward_rows]).square()).sum() / reward_mask.sum()
        else:
            losses['latent'] = ((1 - termination[rows]) * (critic.predictor(features) - next_features[rows]).square().mean(-1)).mean()
        extra['q/mse'] = (critic.hl.to_scalar(logits.detach()) - targets[rows]).square().mean()
        return losses, extra
    compiled_losses = graph_compile(loss_terms)
    results = []
    for batch in range(cli.batches):
        rows = permutation[batch * cli.rows:(batch + 1) * cli.rows]
        losses, extra = compiled_losses(rows)
        metrics = group_gradients(critic, losses, coefficients)
        metrics.update({name: float(value.detach()) for name, value in extra.items()})
        metrics['total/clip_multiplier'] = min(1., args.max_grad_norm / (metrics['total/grad_norm'] + 1e-6))
        results.append(metrics)
    summary = {name: {'mean': float(np.mean([row[name] for row in results])),
                       'sd_across_minibatches': float(np.std([row[name] for row in results], ddof=1)) if cli.batches > 1 else None}
               for name in results[0]}
    report = {'snapshot': str(cli.snapshot.resolve()), 'dataset': str(cli.dataset.resolve()),
              'trainer': trainer_file, 'global_step': int(snapshot['global_step']),
              'rows_per_batch': cli.rows, 'batches': cli.batches, 'seed': cli.seed,
              'notes': ['Frozen snapshot gradients on disjoint subsets of one fresh oracle rollout, not training gradient history.',
                        'Targets are expected-soft critic-bootstrapped returns, not independent simulator labels.',
                        'Within-rollout minibatch SD is not uncertainty across independent runs.',
                        'Target encoder equals current snapshot; approximates first update after target refresh.',
                        'Zero-gradient objective/group pairs have cosine zero by convention.'],
              'coefficients': coefficients, 'summary': summary, 'minibatches': results}
    output = cli.out or cli.dataset.with_name(cli.dataset.stem + '_gradients.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    source_exp = cli.snapshot.parent.name.split('__')[1]
    run_name = f'{args.env_id}__reppo_snapshot_gradients_{source_exp}_{snapshot["global_step"]}__{cli.seed}__{int(time.time())}'
    with SummaryWriter(str(REPO / 'runs' / run_name)) as writer:
        writer.add_text('provenance', json.dumps({k: v for k, v in report.items() if k not in ('summary', 'minibatches')}))
        for name, stats in summary.items():
            writer.add_scalar(f'diagnostics/{name}', stats['mean'], snapshot['global_step'])
    print(json.dumps({'report': str(output), 'tensorboard_run': run_name, 'summary': summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, nargs='?')
    parser.add_argument('dataset', type=Path, nargs='?')
    parser.add_argument('--run-pattern')
    parser.add_argument('--snapshot-steps', type=int, nargs='+', default=[8000000, 12000000, 16000000])
    parser.add_argument('--dataset-dir', type=Path, default=REPO / 'runs/reppo_latentroll_oracle')
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--trainer', type=Path)
    cli = parser.parse_args()
    if cli.run_pattern:
        if cli.snapshot or cli.dataset or cli.out:
            parser.error('--run-pattern cannot be combined with snapshot/dataset/--out')
        matches = [Path(path) for path in glob.glob(str(REPO / 'runs' / cli.run_pattern))
                   if Path(path).is_dir() and list(Path(path).glob('snapshot_*.pt'))]
        if not matches:
            parser.error('No matching run with snapshots')
        run_dir = max(matches, key=lambda path: int(path.name.rsplit('__', 1)[-1]))
        snapshots = sorted(run_dir.glob('snapshot_*.pt'), key=lambda path: int(path.stem.split('_')[-1]))
        for requested in cli.snapshot_steps:
            snapshot = min(snapshots, key=lambda path: abs(int(path.stem.split('_')[-1]) - requested))
            saved = torch.load(snapshot, map_location='cpu', weights_only=False)
            actual = int(saved['global_step'])
            if not requested <= actual < requested + int(saved['args']['batch_size']):
                parser.error(f'No snapshot within one rollout following requested {requested}')
            dataset = cli.dataset_dir / f'{run_dir.name}_{actual}.pt'
            if not dataset.is_file():
                parser.error(f'Missing oracle dataset: {dataset}')
            run(SimpleNamespace(**{**vars(cli), 'snapshot': snapshot, 'dataset': dataset}))
    else:
        if cli.snapshot is None or cli.dataset is None:
            parser.error('Supply snapshot and dataset, or --run-pattern')
        run(cli)


if __name__ == '__main__':
    main()

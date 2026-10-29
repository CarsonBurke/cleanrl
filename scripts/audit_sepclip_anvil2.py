"""Matched-rate real-history frontier, not fresh-training return evidence.

All collection and replay uses CUDA under mlq. New ANVIL states are explicitly
cold controls; only NorMuon history can be restored from old checkpoints.
"""
import argparse
import copy
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
from cleanrl import ppo_continuous_action_sepclip_anvil2_v4 as m


def restore_normuon(opt, saved):
    # Parameter group order and shape banks match v3. Old checkpoint is row/long
    # compatible only on identical lanes; refuse accidental reinterpretation.
    states = list(saved['state'].values())
    assert len(states) == len(opt.stacks)
    for (_, _, state, _), source in zip(opt.stacks, states):
        assert state['velocity'][0].shape == source['momentum_buffer'].shape
        assert state['second'].shape == source['second'].shape
        state['velocity'][0].copy_(source['momentum_buffer'])
        state['second'].copy_(source['second'])


def replay(path, writer):
    snapshot = torch.load(path, map_location='cuda', weights_only=False)
    args = SimpleNamespace(**snapshot['args'])
    assert args.normuon_network == 'actor' and args.num_minibatches == 1
    assert args.target_kl is None
    args.matrix_method, args.matrix_precision = 'normuon', 'bf16'
    args.num_iterations = (args.total_timesteps - 16000) // (args.num_steps * args.num_envs)
    envs = m.make_training_env(args, 'sepclip_anvil2_audit')
    try:
        agent = m.Agent(envs).cuda()
    finally:
        envs.close()
    agent.load_state_dict(snapshot['model'])
    base = copy.deepcopy(agent.state_dict())
    train = snapshot['batch']
    with torch.no_grad():
        train['alpha'], train['beta'], _ = agent.get_policy_and_value(train['obs'])
        train['advantages_mc'] = train['advantages']
        chunks, returns = bench.collect(m, agent, snapshot, 3, 'cuda', 1)
        for chunk in chunks:
            chunk['values'] = agent.get_value(chunk['obs']).flatten()
            chunk['returns'] = chunk['values'] + chunk['advantages']
    hold = bench.concatenate(chunks)
    loss = torch.compile(lambda obs, native, old, adv, ret, values:
        m.ppo_loss(agent, obs, native, old, adv, ret, values, args),
        fullgraph=True, options={'triton.cudagraphs': False})
    cases = []
    # Rate multipliers are relative to the checkpoint's actual annealed LR.
    current_lr = float(snapshot['matrix_optimizer']['param_groups'][0]['lr'])
    configs = [('normuon', 'warm', '.95', 'bf16', x) for x in (.25, .5, 1., 2., 4.)]
    configs += [('normuon', 'warm', '.85', 'bf16', x) for x in (.5, 1.)]
    configs += [(method, 'cold', '.95', precision, x)
                for method, precision in [('normuon', 'bf16'), ('anvil2', 'bf16'), ('anvil2', 'fp32')]
                for x in (.5, 1., 2., 4.)]
    for method, history, momentum, precision, multiplier in configs:
        agent.load_state_dict(base)
        args.matrix_method, args.matrix_precision = method, precision
        args.normuon_momentum = float(momentum)
        args.normuon_beta2 = .9 if method == 'anvil2' else .95
        opt, adam = m.make_optimizers(agent, args)
        opt.set_lr(current_lr * multiplier)
        adam.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        if history == 'warm':
            restore_normuon(opt, snapshot['matrix_optimizer'])
        # For cold ANVIL at late policy age, compare mature rail coefficients,
        # with explicitly reset states. This is not an ANVIL history replay.
        if method == 'anvil2':
            for group in opt.param_groups:
                group['steps'] = ((snapshot['global_step'] - 16000) // args.batch_size - 1) * args.update_epochs
        epochs = []
        for epoch in range(args.update_epochs):
            probe.gradients(agent, train, loss)
            torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()), args.max_grad_norm, foreach=True)
            torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()), args.max_grad_norm, foreach=True)
            opt.step()
            adam.step()
            agent.normalize_matrices()
            train_metrics = probe.evaluate(agent, train)
            # The saved rollout has GAE targets, not lambda=1 targets.
            train_metrics.pop('mc_gain')
            epochs.append(dict(epoch=epoch+1, train=train_metrics, hold=probe.evaluate(agent, hold)))
        label = f'{method}_{history}_m{momentum}_{precision}_lr{multiplier}'
        for tag in ('exact_kl', 'surrogate_gain', 'mc_gain'):
            writer.add_scalar(f'{snapshot["global_step"]}/{label}/{tag}', epochs[-1]['hold'][tag], 10)
        row = dict(label=label, matrix_lr=current_lr*multiplier, epochs=epochs,
                   heldout_chunks=[probe.evaluate(agent, chunk) for chunk in chunks])
        cases.append(row)
        print(json.dumps(dict(label=label, final=epochs[-1])), flush=True)
    return dict(snapshot=str(path), global_step=snapshot['global_step'], cases=cases, rollout_returns=returns,
                limits=['Local policy updates do not establish fresh training performance.',
                        'ANVIL and cold NorMuon reset optimizer history; warm NorMuon retains actual history.',
                        'Warm momentum=.85 is a switch from stored momentum=.95 history, not historical .85 training.',
                        'Heldout chunks are disjoint trajectories, not independent seeds.',
                        'Training gains use GAE; only heldout mc_gain uses lambda=1 targets with critic bootstraps.',
                        'Collection resets saved normalizer state per chunk, then updates it during the rollout.',
                        'Diagnostic snapshots capture rollout-end policy before that iteration\'s PPO update.',
                        'Rate frontier includes simultaneous fixed-LR Adam heads; compare actual measured KL.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshots', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    cli = parser.parse_args()
    m.configure_runtime(matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)
    torch._dynamo.config.recompile_limit = 64
    result = dict(snapshots=[])
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(f'runs/HalfCheetah-v4__sepclip_anvil2_frontier_v4__1__{int(time.time())}') as writer:
        for path in cli.snapshots:
            result['snapshots'].append(replay(path, writer))
            cli.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

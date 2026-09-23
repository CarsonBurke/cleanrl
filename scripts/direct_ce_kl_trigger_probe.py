"""Saved-batch trigger interventions at unchanged LR; no environment training.

Run through mlq. Restores exact model/Adam states and saved minibatch order.
Interventions identify mechanisms, not proposed clipping or momentum recipes.
"""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.direct_ce_kl_diagnostic import BATCH_LOCALS, TRAINER_MODULE, beta_kl, floats, owned


def probe(path, trigger):
    trainer = importlib.import_module(TRAINER_MODULE)
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    args = trainer.validate_args(trainer.Args(**snapshot['args']))
    if not torch.cuda.is_available() or not args.compile or args.norm_adv or args.grad_clip != 'none':
        raise RuntimeError('Requires captured compiled CUDA, unclipped, non-normalized-advantage run')
    trainer.configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    state = snapshot['model_state']
    envs = SimpleNamespace(
        single_observation_space=trainer.gym.spaces.Box(-np.inf, np.inf, shape=(snapshot['observations'].shape[-1],), dtype=np.float32),
        single_action_space=trainer.gym.spaces.Box(state['action_low'].numpy(), state['action_high'].numpy(), dtype=np.float32))
    agent = trainer.Agent(envs, args).cuda()
    optimizer = torch.optim.Adam(tuple(agent.parameters()), lr=args.learning_rate, eps=1e-5, fused=True)
    batch = {key: snapshot[key].cuda() for key in BATCH_LOCALS}
    indices_list = [indices.cuda() for indices in snapshot['minibatch_indices']]
    actor_parameters = tuple(agent.actor.parameters())

    def loss_fn(obs, native, logprob, adv, targets, alpha, beta):
        return trainer.ppo_loss(agent, obs, native, logprob, adv, targets, args, alpha, beta)
    loss_fn = torch.compile(loss_fn, mode=args.compile_mode, fullgraph=True, dynamic=False)
    normalize = torch.compile(agent.normalize_matrices, fullgraph=True, options={'triton.cudagraphs': False})

    @torch.no_grad()
    def evaluate(obs, native, logprob, adv, old_alpha, old_beta):
        alpha, beta = (F.softplus(agent.actor(obs)) + 1).chunk(2, -1)
        ratio = (agent.action_logprob(alpha, beta, native) - logprob).exp()
        active = ((adv > 0) & (ratio <= 1 + args.clip_coef_upper)) | ((adv < 0) & (ratio >= 1 - args.clip_coef))
        weight = adv.abs() * ratio * active
        negative_high = (adv < 0) & (ratio > 1 + args.clip_coef_upper)
        order = weight.argsort(descending=True)
        score_tail = torch.zeros_like(negative_high).scatter(0, order[:(adv.numel()+99)//100], True)
        pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(1-args.clip_coef, 1+args.clip_coef_upper)).mean()
        kl = beta_kl(old_alpha, old_beta, alpha, beta)
        return {'kl': kl.mean(), 'policy_loss': pg, 'ratio_max': ratio.max(),
                'negative_high_fraction': negative_high.float().mean(),
                'negative_high_weight_fraction': torch.where(negative_high, weight, 0).sum() / weight.sum().clamp_min(1e-30)}, negative_high, score_tail
    # Avoid dynamic Boolean indexing in the compiled metric reduction.
    evaluate = torch.compile(evaluate, fullgraph=True, options={'triton.cudagraphs': False})
    def measure():
        return evaluate(*(batch[k] for k in ('observations','native_actions','old_logprobs','advantages','old_alpha','old_beta')))

    prefix = Path(path).with_suffix('')
    output_path = Path(str(prefix) + '_trigger.json')
    writer = SummaryWriter(str(prefix) + '_trigger_tb')
    variants = ('baseline', 'remove_negative_high_once', 'only_negative_high_once',
                'remove_score_tail_once', 'clear_history_at_trigger',
                'clear_moment_after_trigger', 'coast_after_trigger',
                'remove_negative_high_adaptive')
    result = {'snapshot':str(path), 'step':snapshot['step'], 'trigger_update_zero_based':trigger,
              'saved_learning_rate':snapshot['optimizer_state']['param_groups'][0]['lr'],
              'scope':'Exact pre-trigger state interventions, saved minibatches, unchanged LR; no learning benchmark.', 'variants':{}}
    trigger_state = None
    try:
        for variant in variants:
            if variant == 'baseline':
                agent.load_state_dict(state)
                optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer_state']))
                start = 0
            else:
                assert trigger_state is not None
                agent.load_state_dict(trigger_state['model'])
                optimizer.load_state_dict(owned(trigger_state['optimizer']))
                start = trigger
            optimizer.zero_grad(set_to_none=True)
            records = []
            for update in range(start, len(indices_list)):
                torch.compiler.cudagraph_mark_step_begin()
                if variant == 'baseline' and update == trigger:
                    trigger_state = {'model':owned(agent.state_dict()),'optimizer':owned(optimizer.state_dict())}
                pre, negative_high, score_tail = measure()
                pre_numbers = floats(pre)
                advantages = batch['advantages']
                intervention = update == trigger
                if (intervention and variant == 'remove_negative_high_once') or variant == 'remove_negative_high_adaptive':
                    advantages = advantages.masked_fill(negative_high, 0)
                elif intervention and variant == 'only_negative_high_once':
                    advantages = advantages.masked_fill(~negative_high, 0)
                elif intervention and variant == 'remove_score_tail_once':
                    advantages = advantages.masked_fill(score_tail, 0)
                if intervention and variant == 'clear_history_at_trigger':
                    for p in actor_parameters:
                        optimizer.state[p]['exp_avg'].zero_()
                index = indices_list[update]
                loss, _ = loss_fn(batch['observations'][index],batch['native_actions'][index],batch['old_logprobs'][index],
                               advantages[index],batch['target_probs'][index],batch['old_alpha'][index],batch['old_beta'][index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if variant == 'coast_after_trigger' and update > trigger:
                    for p in actor_parameters:
                        p.grad.zero_()
                before = [p.detach().clone() for p in actor_parameters]
                grad_norm = torch.stack([p.grad.square().sum() for p in actor_parameters]).sum().sqrt()
                optimizer.step()
                normalize()
                if variant == 'clear_moment_after_trigger' and update == trigger:
                    for p in actor_parameters:
                        optimizer.state[p]['exp_avg'].zero_()
                post, _, _ = measure()
                post_numbers = floats(post)
                delta = torch.stack([(p.detach()-old).square().sum() for p,old in zip(actor_parameters,before)]).sum().sqrt()
                record = {'update':update,'pre':pre_numbers,'post':post_numbers,
                          'gradient_norm':float(grad_norm),'actor_delta_norm':float(delta)}
                records.append(record)
                writer.add_scalar(variant+'/kl',post_numbers['kl'],update)
            error = None
            if variant == 'baseline':
                error = max(float((v-snapshot['post_model_state'][k].to(v.device)).abs().max()) for k,v in agent.state_dict().items())
                if error != 0:
                    raise RuntimeError(f'Baseline not exact: max parameter error {error}')
            result['variants'][variant] = {'updates':records,'baseline_max_parameter_error':error}
            output_path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'step':snapshot['step'],'variant':variant,'first_kl':records[0]['post']['kl'],'final_kl':records[-1]['post']['kl'],'baseline_error':error}),flush=True)
    finally:
        writer.close()
    print('OUTPUT '+str(output_path),flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: direct_ce_kl_trigger_probe.py SNAPSHOT.pt TRIGGER_UPDATE_ZERO_BASED')
    probe(Path(sys.argv[1]), int(sys.argv[2]))

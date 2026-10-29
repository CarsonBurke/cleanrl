"""Warm matrix-only policy-budget diagnostics, executed only through mlq."""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import awr_update_bench as bench
import diagnose_sepclip_normuon as probe
from audit_sepclip_mechanisms import local_policy_geometry
from cleanrl import ppo_continuous_action_sepclip_policy_budget_anvil_v11 as trainer
from cleanrl.shared import runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    cli = parser.parse_args()
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    saved = torch.load(cli.snapshot, map_location='cuda', weights_only=False)
    args = SimpleNamespace(**saved['args'])
    args.spectral_power = .5
    args.policy_calibration = True
    args.batch_size = args.num_envs*args.num_steps
    args.minibatch_size = args.batch_size
    args.num_iterations = (args.total_timesteps-16000)//args.batch_size
    envs = trainer.make_training_env(args, 'sepclip_mechanisms_fractional_audit')
    try:
        agent = trainer.Agent(envs).cuda()
    finally:
        envs.close()
    agent.load_state_dict(saved['model'])
    base = copy.deepcopy(agent.state_dict())
    train = saved['batch']
    train['advantages_mc'] = train['advantages']  # Only heldout records expose lambda1.
    with torch.no_grad():
        train['alpha'],train['beta'],_ = agent.get_policy_and_value(train['obs'])
        chunks,returns = bench.collect(trainer,agent,saved,3,'cuda',1)
        for chunk in chunks:
            chunk['values'] = agent.get_value(chunk['obs']).flatten()
            chunk['returns'] = chunk['values']+chunk['advantages']
    loss = torch.compile(lambda *fields: trainer.ppo_loss(agent,*fields,args),fullgraph=True,
                         options={'triton.cudagraphs':False})
    held = bench.concatenate(chunks)
    held_gradients = probe.gradients(agent,held,loss)
    actor_logits = torch.compile(agent.actor,fullgraph=True,options={'triton.cudagraphs':False})
    results = []
    names = {p:name for name,p in agent.named_parameters()}
    for power in (0.,.5,1.):
        # Candidates are independent optimizer configurations; isolate the
        # compiler cache rather than exhausting its per-code specialization cap.
        torch.compiler.reset()
        agent.load_state_dict(base)
        args.spectral_power = power
        optimizer,_ = trainer.make_optimizers(agent,args)
        state = copy.deepcopy(saved['matrix_optimizer'])
        for group in state['param_groups']:
            group['spectral_power'] = power
        if isinstance(optimizer,trainer.PolicyCalibratedMatrices):
            shadow = copy.deepcopy(state)
            for group in shadow['param_groups']:
                group['spectral_power'] = 0
            optimizer.load_state_dict(dict(actual=state,shadow=shadow,axes=tuple(optimizer.axes),
                                           metrics=torch.zeros_like(optimizer.metrics)))
        else:
            optimizer.load_state_dict(state)
        optimizer.begin_rollout(args.update_epochs)
        epochs = []
        for epoch in range(4):
            before = {p:p.detach().clone() for group in optimizer.param_groups for p in group['params']}
            with torch.no_grad():
                before_logits = [agent.actor(chunk['obs']).clone() for chunk in chunks]
            probe.gradients(agent,train,loss)
            torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()),args.max_grad_norm,foreach=True)
            if isinstance(optimizer,trainer.PolicyCalibratedMatrices):
                optimizer.step(actor_logits,train['obs'])
            else:
                optimizer.step()
            matrix_rows = []
            with torch.no_grad():
                # Compare the same post-retraction stage for plain and calibrated paths.
                for p in before:
                    axis = 0 if '.down.' in names[p] else 1
                    p.div_(p.norm(dim=axis,keepdim=True))
                for p,old in before.items():
                    axis = 0 if '.down.' in names[p] else 1
                    change = p-old
                    radial = old*(old*change).sum(axis,keepdim=True)
                    tangent = change-radial
                    gradient = held_gradients[names[p]]
                    # Descent direction is minus the observed parameter change.
                    cosine = lambda x: float((-x*gradient).sum()/(x.norm()*gradient.norm()).clamp_min(1e-30))
                    matrix_rows.append(dict(parameter=names[p],radial_fraction=float(radial.norm()/change.norm().clamp_min(1e-30)),
                                            parameter_step_norm=float(change.norm()),base_heldout_cosine=cosine(change),
                                            tangent_base_heldout_cosine=cosine(tangent)))
                chunk_rows = []
                for index,chunk in enumerate(chunks):
                    after_logits = agent.actor(chunk['obs'])
                    geometry = local_policy_geometry(before_logits[index],after_logits,chunk,agent.log_action_scale)
                    geometry['directional_efficiency'] = geometry['first_order_gae_gain']/max(
                        (2*geometry['sample_fisher_quadratic'])**.5,1e-30)
                    metric = probe.evaluate(agent,chunk)
                    metric['critic_bootstrapped_lambda1_gain'] = metric.pop('mc_gain')
                    chunk_rows.append(dict(chunk=index,geometry=geometry,total_from_old=metric))
            epochs.append(dict(epoch=epoch+1,matrices=matrix_rows,chunks=chunk_rows,
                policy_budget_metrics=optimizer.metrics.cpu().tolist()
                    if isinstance(optimizer,trainer.PolicyCalibratedMatrices) else None))
        results.append(dict(power=power,epochs=epochs))
        print(json.dumps({'power':power,'first_epoch_efficiency':[r['geometry']['directional_efficiency']
                           for r in epochs[0]['chunks']]}),flush=True)
    cli.out.parent.mkdir(parents=True,exist_ok=True)
    cli.out.write_text(json.dumps(dict(snapshot=str(cli.snapshot),cases=results,rollout_returns=returns,
        limitations=['Warm identical rails/lane moments; matrix-only replay, head and auxiliary parameters frozen.',
                     'Three correlated heldout chunks at one selected checkpoint, not seeds.',
                     'Collection resets normalizers per chunk and updates within each chunk.',
                     'Matrix cosines in all epochs reference the base heldout gradient; policy-space geometry is current.',
                     'Matrix changes/cosines/radial fractions are measured after the same common retraction stage.',
                     'Lambda1 gain bootstraps critic at boundaries; not environment benchmark return.',
                     'Equal parameter Frobenius norms do not guarantee equal policy Fisher displacement.',
                     'Policy-budget calibration uses training states only and matrix-only displacement, not full actor trust region.',
                     'Shadow rails/lane moments are copied from the good rollout-clock checkpoint; new histories evolve per candidate.',
                     'Positive Gram reweighting retains baseline cascade numerical shape; power1 is only approximately raw.']),indent=2)+'\n')


if __name__ == '__main__':
    main()

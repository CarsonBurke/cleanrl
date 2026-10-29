"""Queue-aware benchmark evidence, using shared TensorBoard readers."""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _runs import RETURN_TAG, RunScalars, find_runs, latest_per_variant


def investigation_report(out, jobs, rows, limitations):
    """Keep source findings separate from CUDA contracts and return evidence."""
    completed = [r for r in rows if r['completed_8m']]
    baseline = next((r for r in completed if r['variant'] == 'normuon_audit_8m_adam'), None)
    families = {}
    for family, prefix in [('anvil2', ('sepclip_anvil2_v4_anvil_', 'sepclip_anvil2_fasttracking_v5_')),
                           ('normuon', 'sepclip_anvil2_v4_normuon_')]:
        candidates = [r for r in completed if r['variant'].startswith(prefix)]
        best = max(candidates, key=lambda r: r['final_100_return'], default=None)
        families[family] = dict(best_completed=best,
                               gain_vs_adam=best['final_100_return'] - baseline['final_100_return']
                               if best and baseline else None,
                               percent_vs_adam=100 * (best['final_100_return'] / baseline['final_100_return'] - 1)
                               if best and baseline else None)
    frontier_path = out / 'frontier.json'
    frontier = json.loads(frontier_path.read_text()) if frontier_path.exists() else None
    if frontier:
        fig, ax = plt.subplots(figsize=(10, 7))
        for snapshot in frontier['snapshots']:
            groups = {}
            for case in snapshot['cases']:
                category, multiplier = case['label'].rsplit('_lr', 1)
                groups.setdefault(category, []).append((case, multiplier))
            for category, cases in groups.items():
                held = [case['epochs'][-1]['hold'] for case, _ in cases]
                display = category
                if category.startswith('anvil2'):
                    display = 'Anvil2 cold, scheduled lookahead, ' + category.rsplit('_', 1)[-1]
                elif '_warm_m.85_' in category:
                    display = 'NorMuon warm .95 history → .85 switch, BF16'
                points = ax.scatter([h['exact_kl'] for h in held], [h['mc_gain'] for h in held],
                                    label=display, marker='o' if '_warm_' in category else 'x')
                for (case, multiplier), h in zip(cases, held):
                    ax.annotate(f'×{multiplier}', (h['exact_kl'], h['mc_gain']), fontsize=7,
                                color=points.get_facecolor()[0], xytext=(3, 3), textcoords='offset points')
        ax.axhline(0, color='gray', linewidth=1)
        ax.set(xlabel='Measured heldout exact policy KL', ylabel='Heldout lambda=1 surrogate gain',
               title='Frozen 4M policy: rate multipliers; circles warm / crosses cold')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / 'frontier.png', dpi=160)
        plt.close(fig)
    contract_path = out / 'contracts.json'
    by_variant = {r['variant']: r for r in rows}
    def contrast(left, right):
        a, b = by_variant.get(left), by_variant.get(right)
        return dict(left=a, right=b, final_return_difference=
                    a['final_100_return'] - b['final_100_return']
                    if a and b and a['completed_8m'] and b['completed_8m'] else None)
    terminal = {'succeeded', 'failed', 'cancelled', 'lost', 'skipped'}
    cull_decisions = []
    for log in (out / 'supervisor.log', Path('benchmarks/sepclip_anvil2_fasttracking_v5/supervisor.log')):
        if not log.exists():
            continue
        lines = log.read_text().splitlines()
        for index, line in enumerate(lines):
            match = re.search(r'\[(\d+)\] (BEHIND|PLATEAU|DEAD|NAN|STALLED)\s+(\S+)', line)
            if match:
                job_id, verdict, variant = match.groups()
                row = by_variant.get(variant)
                cull_decisions.append(dict(job_id=int(job_id), variant=variant,
                                           actor='prefix-scoped autocull supervisor via mlq cancel',
                                           verdict=verdict, reason=lines[index + 1].strip(),
                                           evidence_log=str(log),
                                           observed_final_step=row['final_step'] if row else None,
                                           latest_observed_100_return=row['latest_observed_100_return'] if row else None,
                                           final_8m_score=None))
    report = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status='experiments_terminal' if all(j['state'] in terminal for j in jobs) else 'incomplete',
        scope=dict(env='HalfCheetah-v4', seed=1, requested_steps=8000000,
                   actual_budget=7995008, num_envs=16, env_threads=2,
                   adam_head_critic_lr=.0024, fresh_training=True, finetuning=False),
        reference_revision='4ea6b937337a4889b8cfe3f38a93d120048d8f71',
        spectral_transfer=dict(anvil_tiny_mode_slope_after_margin=643.133485204832 / 1.05,
                               polar_express_tiny_mode_slope_after_margin=989.4683957361317 / 1.02,
                               interpretation='Linearized at zero singular value, neglecting epsilon and BF16 rounding, for unit-Frobenius input. Weaker weak-mode amplification is filtering, not universally better polar accuracy.'),
        claims=dict(algorithm_optimality_established=False, v3_algebra_bug_established=False,
                    adam_outperformed_by_completed_anvil_in_this_seed=
                    families['anvil2']['gain_vs_adam'] > 0 if families['anvil2']['gain_vs_adam'] is not None else None),
        jobs=jobs, cull_decisions=cull_decisions, source_findings=[
            dict(kind='confirmed_previous_bug', finding='v2 cached optimizer group/state references became stale after load_state_dict.',
                 resolution='v3 repaired rebuilding; v4 independently contracts continuation. This does not explain fresh-run underperformance.'),
            dict(kind='reference_convention_difference', finding='v2 output-row equalizer differs from upstream longer-axis lanes on 43x64 gate/up.',
                 resolution='v3 corrected lanes, without closing the fresh-return gap.'),
            dict(kind='prevented_transfer_pitfall', finding='Literal upstream FP32 grad.float().lerp_ aliases and mutates its gradient input.',
                 resolution='v4 uses out-of-place lerp; upstream BF16 .float() makes a copy. The optimizer stack already separates input storage from p.grad; the direct ownership contract checks the arithmetic API itself.'),
            dict(kind='numerical_port_correction', finding='Anvil first Gram precedes normalization; polynomial and action combine before BF16 rounding.',
                 resolution='v4 reuses normalized first Gram and addmm/baddbmm fusion, with BF16-rounded equalized direction. Numeric equivalence must use recorded contract errors.'),
            dict(kind='deliberate_adaptation', finding='FP32 model parameters serve as master weights; no BF16 mantissa reconstruction is needed.'),
            dict(kind='deliberate_adaptation', finding='No LM base-LR-squared slow-sign-gated decay or blindly copied transposed down-projection role multiplier.'),
            dict(kind='deliberate_adaptation', finding='Twin-rail phase fractions scale 1194 LM steps to actual PPO optimizer steps. Copying fractions is a hypothesis, not proof of optimal scheduling.'),
            dict(kind='evidence_label_correction', finding='Final-budget diagnostic snapshot is pre-update; final model is saved after the last update.'),
            dict(kind='evidence_label_correction', finding='Training frontier gain uses GAE, heldout MC gain uses lambda=1 with bootstraps; warm .85 switches from stored .95 history.'),
            dict(kind='contract_definition_correction', finding='Initial BF16 ideal-spectrum errors were 14–25% on condition-number-1e4 inputs, exceeding an unsupported 12% test threshold.',
                 resolution='Retain distortion and FP64-accumulation sensitivity measurements; require strict FP32 oracle, well-conditioned independent BF16 rounding-boundary arithmetic, and upstream parity on the same ill-conditioned inputs. No production arithmetic was changed.'),
            dict(kind='reference_hardware_launch_limitation', finding='Upstream H100 wide-Gram launch requires 147456 shared bytes, exceeding this GPU\'s 101376-byte limit.',
                 resolution='Tests use unchanged upstream XXT_kernel with smaller documented tiles/stages. This checks hardware-adapted source parity; production v4 already uses native CUDA matmuls.'),
            dict(kind='confirmed_job_environment_bug', finding='Report11559 lacked inherited XDG_RUNTIME_DIR and could not reach the machine-wide mlq socket.',
                 resolution='Final experiment/report submissions explicitly inherit XDG_RUNTIME_DIR; the training wrapper fails fast without it so autocull cannot silently lose queue access.')],
        hypotheses=[
            dict(hypothesis='Actor matrix LR is miscalibrated.', evidence='Fresh NorMuon .0025/.01 and Anvil .0025/.005/.01/.02; only .005 NorMuon existed previously.', result=families),
            dict(hypothesis='Faster gradient tracking rescues PPO matrix optimization.', evidence='Fresh NorMuon momentum .85 plus Anvil scheduled fast/slow rails.', result='Use fresh curves and scores; frozen direction replay alone cannot resolve this.'),
            dict(hypothesis='BF16 cascade discrepancies matter.', evidence='Direct upstream and scalar oracle contracts, fresh Anvil FP32 .005 ablation.', result='Use contract errors and paired BF16/FP32 training scores.'),
            dict(hypothesis='Weak-mode noise amplification hurts heldout updates.', evidence='Earlier raw versus whitened gradient alignment and leading-mode energy concentration.', result='Plausible mechanism; this investigation does not establish causality.')],
        training=rows, optimizer_family_results=families,
        controlled_comparisons=dict(
            normuon_momentum=contrast('sepclip_anvil2_v4_normuon_fast', 'normuon_audit_8m_long'),
            normuon_lower_rate=contrast('sepclip_anvil2_v4_normuon_half', 'normuon_audit_8m_long'),
            anvil_lower_rate=contrast('sepclip_anvil2_v4_anvil_half', 'sepclip_anvil2_v4_anvil_full'),
            anvil_precision=contrast('sepclip_anvil2_v4_anvil_fp32', 'sepclip_anvil2_v4_anvil_full'),
            anvil_fasttracking=contrast('sepclip_anvil2_fasttracking_v5_half', 'sepclip_anvil2_v4_anvil_half')),
        fasttracking_iteration=dict(
            version='sepclip_anvil2_fasttracking_v5',
            hypothesis='NorMuon .85 momentum improvement motivates constant .85 fast EMA/lookahead instead of the transferred LM .85-to-.93 schedule.',
            choice='BF16 and actor matrix LR .0025, best completed v4 Anvil; FP32 .005 was culled after consecutive poor checkpoints.',
            controlled_change='Only header and rail_schedule differ from v4: .85 fast EMA/lookahead at all steps; unchanged .98 slow rail, 514/1194 engagement and .4385 blend.',
            decision_limit='One targeted fresh 8M run; this cannot establish global optimizer optimality or seed robustness.'),
        contracts=json.loads(contract_path.read_text()) if contract_path.exists() else None,
        initial_failed_contracts=json.loads((out / 'contracts_attempt1.json').read_text())
        if (out / 'contracts_attempt1.json').exists() else None,
        second_failed_contracts=json.loads((out / 'contracts_attempt2.json').read_text())
        if (out / 'contracts_attempt2.json').exists() else None,
        frontier=frontier,
        limitations=limitations + [
            'Culling is selective negative evidence; a cancelled run has no observed final 8M score.',
            'Cold Anvil has no actual optimizer history in the v3 replay; warm NorMuon does.',
            'Frontier Anvil labels contain constructor m.95, which is unused: actual replay lookahead is scheduled .93, fast beta .85, slow beta .98, blend .4385/.5615.',
            'Three heldout rollout chunks are not independent seeds; rollout normalization updates within chunks.',
            'No gross v3 NorMuon algebra bug has been established; negative training results must not be presented as a solved implementation problem.'])
    (out / 'investigation_report.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, default=Path('benchmarks/sepclip_anvil2_v4'))
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(subprocess.check_output(['mlq', 'show', str(j), '--json'], text=True))
            for j in dict.fromkeys([11335, 11336, 11337, *cli.jobs])]
    patterns = ['normuon_audit_8m_', 'sepclip_anvil2_v4_', 'sepclip_anvil2_fasttracking_v5_']
    tags = {RETURN_TAG, 'losses/post_final_epoch_exact_kl', 'losses/entropy', 'losses/clipfrac',
            'grad/actor_preclip_norm', 'grad/critic_preclip_norm',
            'policy/concentration', 'policy/mean_step_rms', 'policy/log_concentration_step',
            'charts/interval_SPS'}
    tags.update(f'matrix_last/bank{bank}_{metric}' for bank in range(3)
                for metric in ('grad_momentum_cosine', 'nesterov_norm', 'epsilon_fraction'))
    rows = []
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_prop_cycle(color=list(plt.get_cmap('tab20').colors))
    for run in latest_per_variant(find_runs([Path('runs')], patterns, 'HalfCheetah-v4')):
        scalars = RunScalars(run, tags=tags)
        steps, values = scalars.series(RETURN_TAG)
        job = max((j for j in jobs if j['name'] == scalars.variant),
                  key=lambda j: j['createdAt'], default=None)
        # Historical baselines were verified separately in the previous audit.
        historical = scalars.variant.startswith('normuon_audit_8m_')
        completed = job is not None and job['state'] == 'succeeded' and steps.size and steps[-1] >= 7900000
        metrics = {tag: scalars.latest(tag) for tag in sorted(tags - {RETURN_TAG})}
        rows.append(dict(variant=scalars.variant, run=str(run), job_id=job['id'] if job else None,
                         state=job['state'] if job else 'historical_baseline', final_step=int(steps[-1]) if steps.size else 0,
                         completed_8m=bool(completed), final_100_return=float(values[-100:].mean()) if completed else None,
                         latest_observed_100_return=float(values[-100:].mean()) if steps.size else None,
                         matched_step_return={str(step): float(values[(steps >= step - 200000) & (steps <= step)].mean())
                                              if steps.size and steps[-1] >= step
                                              and ((steps >= step - 200000) & (steps <= step)).any() else None
                                              for step in (1000000, 2000000, 4000000, 6000000, 7900000)},
                         latest_metrics=metrics))
        if steps.size >= 100:
            smooth = np.convolve(values, np.ones(100)/100, mode='valid')
            label = scalars.variant.replace('sepclip_anvil2_v4_', '').replace('normuon_audit_8m_', 'baseline_')
            label = label.replace('sepclip_anvil2_fasttracking_v5_', 'fasttracking_v5_')
            ax.plot(steps[99:]/1e6, smooth, label=label, linewidth=2 if historical else 1.5,
                    linestyle='--' if historical else '-')
    ax.set(xlabel='Environment steps (millions)', ylabel='Return, trailing 100 episodes',
           title='Sepclip actor optimizer transfer — HalfCheetah-v4, seed 1')
    ax.grid(alpha=.2)
    if rows:
        ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(cli.out/'learning_curves.png', dpi=160)
    plt.close(fig)
    completed = sorted([r for r in rows if r['completed_8m']], key=lambda r: r['final_100_return'], reverse=True)
    result = dict(jobs=[{k: j.get(k) for k in ('id', 'name', 'state', 'maxParallelRuns', 'stateReason')} for j in jobs],
                  runs=rows, ranked_completed=[r['variant'] for r in completed],
                  limitations=['One seed; this does not establish robustness across seeds or environments.',
                               'Matched-step returns use the same trailing 200k-step window for every run.',
                               'Culled/failed/queued runs have no final-8M score; their latest returns are not ranked.',
                               'V4 Anvil transfers reference cascade and rail schedule fractions; v5 keeps the cascade and slow-rail engagement/blend but uses constant .85 fast EMA/lookahead. Neither copies language-model decay or role multipliers.'])
    (cli.out/'scores.json').write_text(json.dumps(result, indent=2)+'\n')
    investigation_report(cli.out, jobs, rows, result['limitations'])
    for row in completed:
        print(f'{row["variant"]}: {row["final_100_return"]:.1f}')
    for job in jobs:
        print(f'job {job["id"]}: {job["state"]}')


if __name__ == '__main__':
    main()

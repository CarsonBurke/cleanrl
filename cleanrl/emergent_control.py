"""Unseeded evolutionary discovery of coherent, life-local control computation.

A generic recurrent soft-logic genome is inherited; its writable state is not.
No neural policy, optimizer, estimator, or task-solving circuit is planted.
Run GPU training and evaluation through mlq with --max-parallel-runs 1.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import t as student_t
from torch.utils.tensorboard import SummaryWriter

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.collective_control.evolve import CullState, dump_json, paired_validation
from cleanrl.collective_control.genome import Genome
from cleanrl.collective_control.lifetime import (
    OBSERVATION_CONTRACT, TASK_CONTRACT, LifetimeEvaluator, LifetimeSuite,
    calibrate_adapter, make_suite,
)
from cleanrl.collective_control.variation import propose, same_computation, same_genotype
from cleanrl.shared.runtime import configure_runtime

SCHEMA = 1
ALGORITHM = 'emergent-coherent'
SELECTION_CONTRACT = {
    'organism': 'One complete coherent graph; no averaging of competing organisms.',
    'birth': 'Two point events by default; every eighth proposal is structural-only, never bundled.',
    'positive': 'Proposal rank -> fresh shortlist screen -> independent paired lower bound > 0.',
    'neutral': 'Separate uniform genotype-distinct reservoir; exact all-input computation proof; fair coin only without positive birth.',
    'inheritance': 'Genotype only; writable state resets between lifetimes, persists between support and query.',
    'fitness': 'Raw query episode return; support earns no fitness but all its simulation is counted.',
}


@dataclass
class Config:
    run_dir: str = 'runs/emergent_control'
    seed: int = 1
    initial_nodes: int = 32
    max_nodes: int = 256
    mutation_events: float = 2.0
    candidates: int = 128
    shortlist: int = 4
    proposal_lives: int = 2
    screening_lives: int = 4
    validation_lives: int = 16
    development_lives: int = 64
    final_lives: int = 128
    development_every: int = 5
    generations: int = 10000
    total_transitions: int = 128_000_000
    time_limit_seconds: int = 1800
    plateau_warmup_evaluations: int = 20
    plateau_patience: int = 40
    plateau_material_delta: float = 1.0
    plateau_decay: float = .8
    env_threads: int = 8


def validate_config(config: Config) -> None:
    for name in ('initial_nodes', 'max_nodes', 'candidates', 'shortlist', 'proposal_lives',
                 'screening_lives', 'development_lives', 'development_every', 'env_threads'):
        if getattr(config, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('seed', 'generations', 'total_transitions', 'time_limit_seconds',
                 'plateau_warmup_evaluations', 'plateau_patience'):
        if getattr(config, name) < 0:
            raise ValueError(f'{name} must be nonnegative')
    if config.validation_lives < 2 or config.final_lives < 2:
        raise ValueError('validation and final suites need at least two independent lives')
    if config.max_nodes < config.initial_nodes or config.shortlist > config.candidates:
        raise ValueError('node capacity and candidate count must contain their initial/shortlist sizes')
    for name in ('mutation_events', 'plateau_material_delta'):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 <= config.plateau_decay < 1:
        raise ValueError('plateau_decay must lie in [0,1)')


def stream(seed: int, namespace: int, generation: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, namespace, generation]))


def summarize(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()),
            'sem': float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else None,
            'min': float(values.min()), 'max': float(values.max()), 'values': values.tolist()}


def genotype_id(genome: Genome) -> str:
    return hashlib.sha256(json.dumps(genome.to_json(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *(root / 'cleanrl/collective_control' / name for name in (
        'genome.py', 'control.py', 'evolve.py', 'variation.py', 'lifetime.py')),
        root / 'cleanrl/shared/rollout_graph.py', root / 'cleanrl/shared/mujoco_env.py',
        root / 'cleanrl/shared/mujoco_batch.c']
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def choose_birth(parent, proposals, validation, selected_index, rng):
    """A positive candidate and the neutral reservoir have separate admission paths."""
    if selected_index is not None and validation['accepted']:
        candidate = proposals[selected_index]
        return candidate.genome, 'positive', candidate.operator
    neutral = [p for p in proposals if p.neutral and not same_genotype(parent, p.genome)]
    if neutral:
        candidate = neutral[int(rng.integers(len(neutral)))]
        # Keep the proof at the mutation/admission boundary, not only its cached flag.
        if not same_computation(parent, candidate.genome):
            raise ValueError('neutral proposal changed observable computation')
        if rng.random() < .5:
            return candidate.genome, 'neutral', candidate.operator
    return parent, 'none', None


def load_state(path: Path):
    state = json.loads(path.read_text())
    if state.get('schema') != SCHEMA or state.get('algorithm') != ALGORITHM:
        raise ValueError('checkpoint is not this emergence experiment')
    if state['source_sha256'] != source_hashes():
        raise ValueError('checkpoint source provenance differs; use the frozen matching implementation')
    config = Config(**state['config'])
    validate_config(config)
    adapter = ObservationAdapter.from_json(state['observation_adapter'])
    parent = Genome.from_json(state['genome'])
    parent.validate(6, config.max_nodes, False)
    if len(adapter.mean) != 19 or len(adapter.scale) != 19:
        raise ValueError('checkpoint lacks the 19-channel lifetime interface')
    return state, config, adapter, parent


def save_state(path, config, adapter, parent, champion, generation, cull, transitions,
               node_updates, reason, hashes):
    state = {'schema': SCHEMA, 'algorithm': ALGORITHM, 'config': asdict(config),
             'source_sha256': hashes, 'observation_contract': OBSERVATION_CONTRACT,
             'task_contract': TASK_CONTRACT, 'selection_contract': SELECTION_CONTRACT,
             'observation_adapter': adapter.to_json(), 'genome': parent.to_json(),
             'champion': champion, 'generation': generation, 'cull_state': asdict(cull),
             'evaluation_transitions': transitions, 'node_updates': node_updates,
             'calibration_transition_upper_bound': 4 * 128, 'stop_reason': reason,
             'resumable': True, 'randomness': 'Namespaced generation-indexed NumPy generators; no acquired state inherited.'}
    dump_json(path / 'latest.json', state)
    if champion is not None:
        dump_json(path / 'champion.json', {**state, 'genome': champion['genome'],
                  'generation': champion['generation'], 'resumable': False})


def repair_journal(path: Path, generation: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            break
        if item['generation'] <= generation:
            retained.append(line)
    path.write_text(''.join(line + '\n' for line in retained))


def train(config: Config, resume: Path | None = None) -> Path:
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError('compiled CUDA policies are required; no CPU fallback')
    path = Path(config.run_dir)
    device = torch.device('cuda')
    if resume is None:
        path.mkdir(parents=True, exist_ok=False)
        adapter = calibrate_adapter(config.seed, config.env_threads)
        parent = Genome.random(stream(config.seed, 10, 0), 19, 6, config.initial_nodes, False)
        champion, start_generation, prior_transitions, prior_updates = None, 0, 0, 0
        cull = CullState()
        dump_json(path / 'initial_genome.json', parent.to_json())
    else:
        state, _, adapter, parent = load_state(resume)
        if not state['resumable']:
            raise ValueError('resume requires latest.json, not a selected champion')
        champion = state['champion']
        start_generation = state['generation'] + 1
        prior_transitions, prior_updates = state['evaluation_transitions'], state['node_updates']
        cull = CullState(**state['cull_state'])
        for name in ('metrics.jsonl', 'lineage.jsonl'):
            repair_journal(path / name, state['generation'])
    hashes = source_hashes()
    development = make_suite(config.seed, 31, 0, config.development_lives)
    critical = float(student_t.ppf(.95, config.validation_lives - 1))
    started = time.monotonic()
    with LifetimeEvaluator(adapter, config.max_nodes, device, config.env_threads) as evaluator, \
            SummaryWriter(str(path / 'tb'), purge_step=start_generation if resume else None) as writer, \
            (path / 'metrics.jsonl').open('a') as metrics, (path / 'lineage.jsonl').open('a') as lineage:
        for generation in range(start_generation, config.generations + 1):
            row = {'generation': generation, 'birth': 'none', 'operator': None,
                   'proposals': 0, 'neutral_candidates': 0, 'positive_candidates': 0}
            before_id = genotype_id(parent)
            if generation:
                proposals = propose(parent, stream(config.seed, 101, generation), 19, 6,
                                    config.max_nodes, config.mutation_events, config.candidates)
                positive_indices = [i for i, p in enumerate(proposals) if not p.neutral]
                row.update(proposals=len(proposals), neutral_candidates=sum(p.neutral for p in proposals),
                           positive_candidates=len(positive_indices))
                selected_index, validation = None, {'accepted': False}
                if positive_indices:
                    proposal_suite = make_suite(config.seed, 201, generation, config.proposal_lives)
                    scores = evaluator.evaluate([parent] + [proposals[i].genome for i in positive_indices], proposal_suite).scores
                    gains = (scores[1:] - scores[0]).mean(1)
                    rank = np.argsort(-gains, kind='stable')[:config.shortlist]
                    shortlist = [positive_indices[i] for i in rank]
                    screen_suite = make_suite(config.seed, 202, generation, config.screening_lives)
                    screen = evaluator.evaluate([parent] + [proposals[i].genome for i in shortlist], screen_suite).scores
                    winner = int(np.argmax((screen[1:] - screen[0]).mean(1)))
                    selected_index = shortlist[winner]
                    validation_suite = make_suite(config.seed, 203, generation, config.validation_lives)
                    checked = evaluator.evaluate([parent, proposals[selected_index].genome], validation_suite).scores
                    validation = paired_validation(checked[0], checked[1], critical)
                    row.update(proposal_best_gain=float(gains[rank[0]]),
                               screening_gain=float((screen[winner+1]-screen[0]).mean()),
                               validation=validation)
                parent, birth, operator = choose_birth(parent, proposals, validation, selected_index,
                                                      stream(config.seed, 102, generation))
                row.update(birth=birth, operator=operator)
                if birth != 'none':
                    lineage.write(json.dumps({'generation': generation, 'parent': before_id,
                        'child': genotype_id(parent), 'birth': birth, 'operator': operator,
                        'genome': parent.to_json(), 'validation': validation if birth == 'positive' else None}) + '\n')
                    lineage.flush()

            transitions = prior_transitions + evaluator.evaluation_transitions
            reason = None
            if config.total_transitions and transitions >= config.total_transitions:
                reason = 'transition_budget'
            elif config.time_limit_seconds and time.monotonic()-started >= config.time_limit_seconds:
                reason = 'time_limit'
            elif generation == config.generations:
                reason = 'generation_limit'
            if generation % config.development_every == 0 or reason:
                result = evaluator.evaluate([parent], development)
                values = result.scores[0]
                value = float(values.mean())
                if champion is None or value > champion['mean_return']:
                    champion = {'generation': generation, 'mean_return': value,
                                'returns': values.tolist(), 'genome': parent.to_json()}
                plateau = cull.update(value, config)
                if reason is None and plateau:
                    reason = 'development_plateau'
                row.update(development_query_return=value,
                           development_support_return=float(result.support_returns.mean()),
                           development_query_returns=values.tolist(), champion_return=champion['mean_return'])
                writer.add_scalar('return/development_query', value, generation)
                writer.add_scalar('return/development_support', float(result.support_returns.mean()), generation)
                writer.add_scalar('return/champion_query', champion['mean_return'], generation)
                print(json.dumps({'event': 'development', **row, 'query_returns_omitted': True,
                                  'development_query_returns': None}), flush=True)
            transitions = prior_transitions + evaluator.evaluation_transitions
            updates = prior_updates + evaluator.node_updates
            if reason is None and config.total_transitions and transitions >= config.total_transitions:
                reason = 'transition_budget'
            elif reason is None and config.time_limit_seconds and time.monotonic()-started >= config.time_limit_seconds:
                reason = 'time_limit'
            row.update(evaluation_transitions=transitions, node_updates=updates,
                       nodes=len(parent.nodes), genotype=genotype_id(parent), stop_reason=reason,
                       elapsed_seconds=time.monotonic()-started)
            metrics.write(json.dumps(row)+'\n')
            metrics.flush()
            writer.add_scalar('birth/neutral', int(row['birth']=='neutral'), generation)
            writer.add_scalar('birth/positive', int(row['birth']=='positive'), generation)
            writer.add_scalar('birth/neutral_candidates', row['neutral_candidates'], generation)
            writer.add_scalar('program/nodes', len(parent.nodes), generation)
            writer.add_scalar('resources/transitions', transitions, generation)
            writer.add_scalar('resources/node_updates', updates, generation)
            writer.flush()
            save_state(path, config, adapter, parent, champion, generation, cull, transitions,
                       updates, reason, hashes)
            if reason:
                print(json.dumps({'event': 'AUTOCULL' if reason == 'development_plateau' else 'complete',
                                  'generation': generation, 'reason': reason, 'evaluation_transitions': transitions}), flush=True)
                break
    return path


def evaluate_checkpoint(checkpoint: Path, count: int | None = None) -> dict:
    state, config, adapter, genome = load_state(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('evaluation requires compiled CUDA, never CPU fallback')
    count = config.final_lives if count is None else count
    if count < 2:
        raise ValueError('acquisition measurement requires at least two lives')
    suite = make_suite(config.seed, 7701, state['generation'], count)
    modes = ('intact', 'reset', 'donor', 'recent', 'no_reward')
    summaries, results = {}, {}
    with LifetimeEvaluator(adapter, config.max_nodes, torch.device('cuda'), config.env_threads) as evaluator:
        for mode in modes:
            result = evaluator.evaluate([genome], suite, mode)
            results[mode] = result
            summaries[mode] = {'query': summarize(result.query_returns[0]),
                               'support': summarize(result.support_returns[0]),
                               'first_actions': result.first_actions[0].tolist(),
                               'query_blocks': result.query_blocks[0].tolist()}
        nominal_suite = LifetimeSuite(suite.support_seeds, suite.query_seeds, np.ones(count))
        nominal = evaluator.evaluate([genome], nominal_suite, 'reset')
        nominal_summary = summarize(nominal.scores[0])
        contrasts = {}
        for mode in modes[1:]:
            a, b = results['intact'], results[mode]
            contrasts[mode] = {
                'query_gain': summarize(a.query_returns[0]-b.query_returns[0]),
                'first_100_gain': summarize(a.query_blocks[0, :, 0]-b.query_blocks[0, :, 0]),
                'last_900_gain': summarize(a.query_blocks[0, :, 1:].sum(1)-b.query_blocks[0, :, 1:].sum(1)),
                'first_action_max_difference': float(np.max(np.abs(a.first_actions-b.first_actions)))}
        report = {'schema': SCHEMA, 'algorithm': ALGORITHM, 'checkpoint': str(checkpoint),
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'source_sha256': state['source_sha256'], 'generation': state['generation'],
                  'suite': suite.to_json(), 'task_contract': TASK_CONTRACT,
                  'observation_contract': OBSERVATION_CONTRACT, 'interventions': summaries,
                  'intact_minus': contrasts, 'nominal_cold_query': nominal_summary,
                  'evaluation_transitions': evaluator.evaluation_transitions,
                  'node_updates': evaluator.node_updates, 'replay_transitions': evaluator.replay_transitions,
                  'replay_node_updates': evaluator.replay_node_updates,
                  'interpretation': [
                      'Hidden-gain query return is not standard HalfCheetah benchmark return.',
                      'Nominal cold query uses gain1 and pristine state: actual unchanged HalfCheetah reward.',
                      'No final data selected parents or champion; these are frozen paired interventions.',
                      'State/reset effects alone can be controller phase or warmup, not learning.',
                      'Donor experience is on-policy under the opposite gain; actions/histories can differ.',
                      'Recent8 control is not a universal removal of all immediate-context strategies.',
                      'Null late effects do not exclude rapid reacquisition; feedback ablation can be out of distribution.',
                      'One evolved lineage does not establish evolutionary reliability or a novel optimizer.' ]}
    output = checkpoint.parent / f'acquisition_generation_{state["generation"]}.json'
    dump_json(output, report)
    dump_json(checkpoint.parent / 'final_result.json', report)
    print(json.dumps({'event': 'frozen_acquisition', 'output': str(output),
        'intact_query': summaries['intact']['query']['mean'],
        'intact_minus_reset': contrasts['reset']['query_gain']['mean'],
        'intact_minus_donor': contrasts['donor']['query_gain']['mean'],
        'nominal_cold_query': nominal_summary['mean']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    for name, field in Config.__dataclass_fields__.items():
        training.add_argument('--'+name.replace('_', '-'), type=type(field.default), default=field.default)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--checkpoint', type=Path, required=True)
    evaluation.add_argument('--lives', type=int)
    resume = commands.add_parser('resume')
    resume.add_argument('--run-dir', type=Path, required=True)
    resume.add_argument('--additional-transitions', type=int, required=True)
    args = parser.parse_args()
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    if args.command == 'evaluate':
        evaluate_checkpoint(args.checkpoint, args.lives)
    elif args.command == 'resume':
        checkpoint = args.run_dir / 'latest.json'
        state, config, _, _ = load_state(checkpoint)
        if args.additional_transitions <= 0:
            parser.error('--additional-transitions must be positive')
        config.run_dir = str(args.run_dir)
        config.total_transitions = state['evaluation_transitions'] + args.additional_transitions
        if state['generation'] >= config.generations:
            config.generations = state['generation'] + 10000
        path = train(config, checkpoint)
        evaluate_checkpoint(path / 'champion.json')
    else:
        config = Config(**{k:v for k,v in vars(args).items() if k != 'command'})
        path = train(config)
        evaluate_checkpoint(path / 'champion.json')


if __name__ == '__main__':
    main()

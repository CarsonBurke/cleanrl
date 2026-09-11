"""Evolve independent arithmetic/state programs, then test acquired control.

Generic arithmetic, synchronous state and mutation/selection are supplied physics.
There is no supplied controller, network topology, estimator or inner optimizer.
Nominal locomotion must evolve before selection asks descendants to adapt.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.collective_control.evolve import CullState, dump_json
from cleanrl.evolving_programs.genome import Genome, needed_nodes
from cleanrl.evolving_programs.lifetime import (
    ProgramEvaluator, ProgramSuite, calibrate_adapter, make_suite,
    TASK_CONTRACT, OBSERVATION_CONTRACT,
)
from cleanrl.shared.runtime import configure_runtime

SCHEMA = 1
ALGORITHM = 'arithmetic-population'


@dataclass
class Config:
    run_dir: str = 'runs/arithmetic_evolution'
    seed: int = 1
    population: int = 256
    initial_nodes: int = 16
    max_nodes: int = 128
    ticks: int = 4
    mutation_events: float = 2.0
    copy_probability: float = .1
    rank_temperature: float = .1
    exploration: float = .05
    training_lives: int = 2
    development_lives: int = 64
    development_candidates: int = 8
    development_every: int = 10
    final_lives: int = 128
    competence_gate: float = 1000.0
    generations: int = 2000
    total_transitions: int = 256_000_000
    time_limit_seconds: int = 2400
    plateau_warmup_evaluations: int = 20
    plateau_patience: int = 30
    plateau_material_delta: float = 5.0
    plateau_decay: float = .8
    env_threads: int = 8


def validate_config(config):
    for name in ('population', 'initial_nodes', 'max_nodes', 'ticks', 'training_lives',
                 'development_lives', 'development_candidates', 'development_every', 'env_threads'):
        if getattr(config, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('seed', 'generations', 'total_transitions', 'time_limit_seconds',
                 'plateau_warmup_evaluations', 'plateau_patience'):
        if getattr(config, name) < 0:
            raise ValueError(f'{name} must be nonnegative')
    if config.population < 2 or config.final_lives < 2:
        raise ValueError('need multiple organisms and independent final lives')
    if config.initial_nodes > config.max_nodes or config.development_candidates > config.population:
        raise ValueError('initial/selected counts exceed their capacity')
    for name in ('rank_temperature', 'competence_gate'):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('mutation_events', 'plateau_material_delta'):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('copy_probability', 'exploration'):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f'{name} must lie in [0,1]')
    if not 0 <= config.plateau_decay < 1:
        raise ValueError('invalid plateau decay')


def rng(seed, namespace, generation):
    return np.random.default_rng(np.random.SeedSequence([seed, namespace, generation]))


def rank_probabilities(fitness, temperature=.1, exploration=.05):
    """Average exact tie ranks; score spacing never sharpens selection."""
    fitness = np.asarray(fitness, dtype=np.float64)
    if fitness.ndim != 1 or not len(fitness) or np.any(np.isnan(fitness)) or np.any(np.isposinf(fitness)):
        raise ValueError('fitness must be a vector of finite values or negative infinity')
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= exploration <= 1:
        raise ValueError('invalid rank-selection parameters')
    order = np.argsort(-fitness, kind='stable')
    ranks = np.empty(len(fitness), dtype=np.float64)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and fitness[order[end]] == fitness[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    weights = np.exp(-(ranks-ranks.min())/(temperature*len(fitness)))
    weights /= weights.sum()
    return (1-exploration)*weights + exploration/len(fitness)


def fitness_values(result):
    # One failed life invalidates this organism's fitness on the suite. Never
    # average safe-zero actuator returns into an apparently competent policy.
    return result.scores.mean(axis=1)


def finite_summary(values, valid=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) if valid is None else np.asarray(valid, bool) & np.isfinite(values)
    usable = values[valid]
    complete = bool(np.all(valid))
    return {'mean': float(usable.mean()) if complete else None,
            'sem': float(usable.std(ddof=1)/np.sqrt(len(usable))) if complete and len(usable)>1 else None,
            'valid_lives': int(valid.sum()), 'lives': int(valid.size),
            'values': [float(v) if ok else None for v,ok in zip(values,valid)]}


def identity(genome):
    return hashlib.sha256(json.dumps(genome.to_json(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def genome_delta(parent, child):
    before = {n['node_id']:n for n in parent.to_json()['nodes']}
    after = child.to_json()
    return {'nodes': [n for n in after['nodes'] if before.get(n['node_id']) != n],
            'deleted': sorted(set(before)-{n['node_id'] for n in after['nodes']}),
            'outputs': after['outputs'], 'next_id': after['next_id']}


def breed(population, fitness, config, generation):
    selection = rng(config.seed, 101, generation)
    weights = rank_probabilities(fitness, config.rank_temperature, config.exploration)
    parents = selection.choice(len(population), config.population, p=weights)
    children, births = [], []
    for index, parent_index in enumerate(parents):
        parent = population[int(parent_index)]
        child = parent.clone()
        birth_rng = rng(config.seed, 1000+index, generation)
        if birth_rng.random() < config.copy_probability:
            operator = 'copy'
        else:
            structural = index % 8 == 0
            operator = 'structure' if structural else 'point'
            child.mutate(birth_rng, 19, 6, config.max_nodes,
                         0.0 if structural else config.mutation_events, 1.0 if structural else 0.0)
        children.append(child)
        births.append({'parent_index': int(parent_index), 'parent': identity(parent),
                       'child': identity(child), 'operator': operator, 'delta': genome_delta(parent,child)})
    frequencies = np.bincount(parents, minlength=len(population))/len(parents)
    nonzero = frequencies[frequencies>0]
    return children, births, {'distinct_parents': int(len(nonzero)),
                              'effective_parents': float(np.exp(-np.sum(nonzero*np.log(nonzero))))}


def provenance():
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *(root/'cleanrl/evolving_programs'/name for name in ('genome.py','policy.py','lifetime.py')),
             *(root/'cleanrl/shared'/name for name in ('rollout_graph.py','runtime.py','mujoco_env.py','mujoco_batch.c')),
             *(root/'cleanrl/collective_control'/name for name in ('control.py','lifetime.py','evolve.py'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def save(path, config, adapter, population, generation, profile, champions, cull, counters,
         stage_boundary, reason, hashes):
    data = {'schema':SCHEMA, 'algorithm':ALGORITHM, 'config':asdict(config),
            'source_sha256':hashes, 'task_contract':TASK_CONTRACT,
            'observation_contract':OBSERVATION_CONTRACT, 'adapter':adapter.to_json(),
            'generation':generation, 'profile':profile, 'population':[g.to_json() for g in population],
            'champions':champions, 'cull_state':asdict(cull), 'counters':counters,
            'stage_boundary':stage_boundary, 'stop_reason':reason,
            'inheritance':'Genotypes only; no acquired state, optimizer or hand-built controller is inherited.',
            'reproduction':'Rank-based random parents with full support, exact ties, no privileged elites; independent whole programs.'}
    dump_json(path/'latest.json',data)
    for stage, champion in champions.items():
        if champion:
            dump_json(path/f'champion_{stage}.json',{key:value for key,value in data.items() if key!='population'} |
                      {'champion':champion,'profile':stage,'genome':champion['genome'],
                       'generation':champion['generation']})


def stop_reason(config, generation, transitions, elapsed):
    if config.total_transitions and transitions >= config.total_transitions:
        return 'transition_budget'
    if config.time_limit_seconds and elapsed >= config.time_limit_seconds:
        return 'time_limit'
    if generation >= config.generations:
        return 'generation_limit'
    return None


def train(config):
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError('arithmetic programs require compiled CUDA; no CPU fallback')
    path = Path(config.run_dir)
    path.mkdir(parents=True,exist_ok=False)
    adapter = calibrate_adapter(config.seed,config.env_threads)
    population = [Genome.random(rng(config.seed,10,index),19,6,config.initial_nodes) for index in range(config.population)]
    dump_json(path/'initial_population.json',[g.to_json() for g in population])
    hashes, champions, profile, stage_boundary = provenance(), {'nominal':None,'positive':None}, 'nominal', None
    cull, started = CullState(), time.monotonic()
    with ProgramEvaluator(adapter,config.max_nodes,config.ticks,torch.device('cuda'),config.env_threads) as evaluator, \
         SummaryWriter(str(path/'tb')) as writer, (path/'metrics.jsonl').open('w') as metrics, \
         (path/'lineage.jsonl').open('w') as lineage:
        for generation in range(config.generations+1):
            training = make_suite(config.seed,201,generation,config.training_lives,profile=profile)
            result = evaluator.evaluate(population,training,profile=profile)
            fitness = fitness_values(result)
            finite = np.isfinite(fitness)
            row = {'generation':generation,'profile':profile,'viable_fraction':float(finite.mean()),
                   'training_best':float(fitness[finite].max()) if finite.any() else None,
                   'training_viable_mean':float(fitness[finite].mean()) if finite.any() else None,
                   'mean_nodes':float(np.mean([len(g.nodes) for g in population])),
                   'mean_reachable_nodes':float(np.mean([len(needed_nodes(g)) for g in population]))}
            reason = stop_reason(config,generation,evaluator.evaluation_transitions,time.monotonic()-started)
            stage_qualified = False
            if generation % config.development_every == 0 or reason:
                eligible = np.flatnonzero(finite)
                if len(eligible):
                    order = eligible[np.argsort(-fitness[eligible],kind='stable')[:config.development_candidates]]
                    development = make_suite(config.seed,31,0,config.development_lives,profile=profile)
                    evaluated = evaluator.evaluate([population[int(i)] for i in order],development,profile=profile)
                    values = fitness_values(evaluated)
                    if np.any(np.isfinite(values)):
                        best = int(np.argmax(values))
                        value = float(values[best])
                        champion = champions[profile]
                        if champion is None or value > champion['mean_return']:
                            champions[profile] = {'generation':generation,'mean_return':value,
                                'returns':evaluated.query_returns[best].tolist(),
                                'genome':population[int(order[best])].to_json()}
                        plateau = cull.update(value,config)
                        row.update(development_best=value,champion_return=champions[profile]['mean_return'])
                        writer.add_scalar(f'return/{profile}_development',value,generation)
                        writer.add_scalar(f'return/{profile}_champion',champions[profile]['mean_return'],generation)
                        stage_qualified = profile=='nominal' and value >= config.competence_gate
                        if reason is None and plateau and not stage_qualified:
                            reason='development_plateau'
                print(json.dumps({'event':'development',**row}),flush=True)
            # Development itself can cross a resource boundary.
            reason = stop_reason(config,generation,evaluator.evaluation_transitions,time.monotonic()-started) or reason
            if stage_qualified and stage_boundary is None:
                stage_boundary={'generation':generation,'nominal_development_return':champions['nominal']['mean_return'],
                                'threshold':config.competence_gate,'transitions':evaluator.evaluation_transitions,
                                'ancestor':identity(Genome.from_json(champions['nominal']['genome']))}
                dump_json(path/'nominal_foundation_population.json',[g.to_json() for g in population])
                print(json.dumps({'event':'competence_gate',**stage_boundary}),flush=True)
            counters={name:getattr(evaluator,name) for name in ('evaluation_transitions','logical_node_updates',
                      'capacity_node_updates','replay_transitions','replay_node_updates')}
            row.update(counters,elapsed_seconds=time.monotonic()-started,stop_reason=reason)
            if reason is None:
                children,births,parent_stats=breed(population,fitness,config,generation+1)
                row.update(parent_stats)
                lineage.write(json.dumps({'generation':generation+1,'parent_generation':generation,'profile':profile,'births':births})+'\n')
                lineage.flush()
            metrics.write(json.dumps(row)+'\n'); metrics.flush()
            writer.add_scalar('population/viable_fraction',row['viable_fraction'],generation)
            writer.add_scalar('population/mean_nodes',row['mean_nodes'],generation)
            writer.add_scalar('population/mean_reachable_nodes',row['mean_reachable_nodes'],generation)
            if 'effective_parents' in row:
                writer.add_scalar('selection/effective_parents',row['effective_parents'],generation)
            writer.add_scalar('resources/physics_transitions',evaluator.evaluation_transitions,generation)
            writer.add_scalar('resources/capacity_node_updates',evaluator.capacity_node_updates,generation)
            writer.flush()
            save(path,config,adapter,population,generation,profile,champions,cull,counters,stage_boundary,reason,hashes)
            if reason:
                print(json.dumps({'event':'complete','reason':reason,'generation':generation,
                                  'profile':profile,**counters}),flush=True)
                break
            population=children
            if stage_qualified:
                profile='positive'
                cull=CullState()
    return path


def load_checkpoint(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema')!=SCHEMA or data.get('algorithm')!=ALGORITHM or 'genome' not in data:
        raise ValueError('requires an arithmetic-population champion checkpoint')
    if data['source_sha256']!=provenance():
        raise ValueError('checkpoint requires its matching frozen source')
    config=Config(**data['config']); validate_config(config)
    genome=Genome.from_json(data['genome']); genome.validate(19,6,config.max_nodes)
    return data,config,ObservationAdapter.from_json(data['adapter']),genome


def result_summary(result):
    valid=(result.support_valid & result.query_valid)[0]
    return {'query':finite_summary(result.query_returns[0],valid),
            'support_valid':result.support_valid[0].tolist(),'query_valid':result.query_valid[0].tolist(),
            'first_actions':result.first_actions[0].tolist(), 'query_blocks':result.query_blocks[0].tolist(),
            'nominal_action_rms':float(np.sqrt(result.action_square_sum[0].sum()/(len(valid)*1000*6))),
            'metadata':result.metadata}


def evaluate_checkpoint(path,count=None):
    data,config,adapter,genome=load_checkpoint(path)
    count=config.final_lives if count is None else count
    if count<2: raise ValueError('final evaluation needs independent lives')
    results={}
    # Stage checkpoints are frozen before either sees this common final suite.
    suite=make_suite(config.seed,7701,0,count,profile='positive')
    nominal_suite=ProgramSuite(suite.support_seeds,suite.query_seeds,np.ones(count))
    with ProgramEvaluator(adapter,config.max_nodes,config.ticks,torch.device('cuda'),config.env_threads) as evaluator:
        nominal=evaluator.evaluate([genome],nominal_suite,profile='nominal')
        for mode in ('intact','reset','donor','same_task','recent','no_reward'):
            results[mode]=evaluator.evaluate([genome],suite,profile='positive',intervention=mode)
        contrasts={}
        intact=results['intact']
        for mode,other in results.items():
            if mode=='intact': continue
            valid=(intact.support_valid & intact.query_valid & other.support_valid & other.query_valid)[0]
            contrasts[mode]={'gain':finite_summary(intact.query_returns[0]-other.query_returns[0],valid),
                'first_100_gain':finite_summary(intact.query_blocks[0,:,0]-other.query_blocks[0,:,0],valid),
                'last_900_gain':finite_summary(intact.query_blocks[0,:,1:].sum(1)-other.query_blocks[0,:,1:].sum(1),valid),
                'first_action_max_difference':float(np.max(np.abs(intact.first_actions-other.first_actions)))}
        counters={name:getattr(evaluator,name) for name in ('evaluation_transitions','logical_node_updates',
                  'capacity_node_updates','replay_transitions','replay_node_updates')}
        report={'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'source_sha256':data['source_sha256'],'trained_profile':data['profile'],'generation':data['generation'],
            'nominal':result_summary(nominal),'interventions':{m:result_summary(r) for m,r in results.items()},
            'intact_minus':contrasts,'suite':suite.to_json(),'counters':counters,
            'caveats':['A high return is evolved control, not evidence of an acquired learner.',
                'Matched-positive versus mismatched-positive support is primary context-specific state evidence; cold-state differences alone can be phase/warmup.',
                'Same-task support uses another reset seed; donor changes support trajectories as well as context.',
                'Recent replay and no-reward are interventions, not universal removal of all immediate-context or learning mechanisms.',
                'Null late differences do not exclude rapid reacquisition; useful state would still need causal mechanism analysis.',
                'Invalid programs receive no eligible score; reported actuator RMS includes safe-zero actions of invalid lanes.',
                'Development selected the champion; frozen final suites were not used for reproduction or stage changes.',
                'Primitive arithmetic, source interfaces, four internal ticks and the staged task are explicit supplied priors, not inventions.']}
    output=Path(path).parent/f'final_{data["profile"]}.json'; dump_json(output,report)
    print(json.dumps({'event':'frozen_evaluation','output':str(output),'nominal':report['nominal']['query'],
                      'positive':report['interventions']['intact']['query'],
                      'donor_gain':contrasts['donor']['gain']}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    training=commands.add_parser('train')
    for name,field in Config.__dataclass_fields__.items():
        training.add_argument('--'+name.replace('_','-'),type=type(field.default),default=field.default)
    evaluation=commands.add_parser('evaluate'); evaluation.add_argument('--checkpoint',type=Path,required=True)
    evaluation.add_argument('--lives',type=int)
    args=parser.parse_args(); configure_runtime(matmul_precision='highest',allow_tf32=False)
    if args.command=='evaluate': evaluate_checkpoint(args.checkpoint,args.lives)
    else:
        config=Config(**{k:v for k,v in vars(args).items() if k!='command'})
        path=train(config)
        checkpoints=[path/'champion_nominal.json',path/'champion_positive.json']
        for checkpoint in checkpoints:
            if checkpoint.exists(): evaluate_checkpoint(checkpoint)
        if not any(p.exists() for p in checkpoints):
            raise RuntimeError('no fully viable development champion; population/checkpoints retained, no competent fallback supplied')


if __name__=='__main__': main()

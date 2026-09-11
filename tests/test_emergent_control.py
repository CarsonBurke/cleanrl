import json

import numpy as np
import pytest

from cleanrl.collective_control.genome import Genome, Node, Source
from cleanrl.collective_control.variation import Proposal
from cleanrl.emergent_control import (
    ALGORITHM, SCHEMA, choose_birth, genotype_id, load_state, repair_journal, source_hashes,
)


class Coin:
    def __init__(self, value):
        self.value = value

    def integers(self, count):
        return count - 1

    def random(self):
        return self.value


def genome():
    return Genome([Node(0, (Source(0), Source(3)), np.array([0, 0, 1, 1], np.float32), .1, .5)],
                  [0] * 6, 1)


def offspring(parent):
    child = parent.clone()
    child.nodes.append(Node(1, (Source(1, 0), Source(3)), np.full(4, .5, np.float32), .1, .5))
    child.next_id = 2
    return child


def test_neutral_capacity_survives_without_positive_reward_gain():
    parent = genome()
    child = offspring(parent)
    result, kind, operator = choose_birth(parent, [Proposal(child, 'structure', True)],
                                          {'accepted': False}, None, Coin(.25))
    assert result is child and kind == 'neutral' and operator == 'structure'
    assert len(result.nodes) > len(parent.nodes)
    rejected, kind, _ = choose_birth(parent, [Proposal(child, 'structure', True)],
                                     {'accepted': False}, None, Coin(.75))
    assert rejected is parent and kind == 'none'


def test_positive_birth_has_precedence_over_separate_neutral_reservoir():
    parent = genome()
    neutral = offspring(parent)
    positive = parent.clone()
    positive.nodes[0].q[0] = .2
    choices = [Proposal(neutral, 'structure', True), Proposal(positive, 'point', False)]
    accepted, kind, _ = choose_birth(parent, choices, {'accepted': True}, 1, Coin(.25))
    assert accepted is positive and kind == 'positive'
    accepted, kind, _ = choose_birth(parent, choices, {'accepted': False}, 1, Coin(.25))
    assert accepted is neutral and kind == 'neutral'


def test_equal_fitness_alone_does_not_authorize_behavioral_drift():
    parent = genome()
    changed = parent.clone()
    changed.nodes[0].initial = .2
    accepted, kind, _ = choose_birth(parent, [Proposal(changed, 'point', False)],
                                     {'accepted': False, 'paired_mean_gain': 0}, 0, Coin(.25))
    assert accepted is parent and kind == 'none'
    with pytest.raises(ValueError, match='observable computation'):
        choose_birth(parent, [Proposal(changed, 'point', True)], {'accepted': False}, None, Coin(.25))


def test_neutral_parent_clones_do_not_consume_admission():
    parent = genome()
    accepted, kind, _ = choose_birth(parent, [Proposal(parent.clone(), 'point', True)],
                                     {'accepted': False}, None, Coin(.25))
    assert accepted is parent and kind == 'none'
    assert genotype_id(parent) != genotype_id(offspring(parent))


def test_resume_repairs_partial_and_beyond_checkpoint_lineage(tmp_path):
    journal = tmp_path / 'lineage.jsonl'
    journal.write_text('{"generation": 1, "child": "kept"}\n'
                       '{"generation": 2, "child": "uncommitted"}\n{"generation":')
    repair_journal(journal, 1)
    assert [json.loads(line)['child'] for line in journal.read_text().splitlines()] == ['kept']


def test_checkpoint_rejects_incompatible_execution_before_replay(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    hashes = source_hashes()
    hashes['cleanrl/collective_control/control.py'] = 'different interpreter'
    checkpoint.write_text(json.dumps({'schema': SCHEMA, 'algorithm': ALGORITHM, 'source_sha256': hashes}))
    with pytest.raises(ValueError, match='provenance'):
        load_state(checkpoint)


def test_development_exhausting_time_budget_stops_before_another_generation(monkeypatch, tmp_path):
    from cleanrl import emergent_control as experiment
    from cleanrl.collective_control.control import ObservationAdapter
    from cleanrl.collective_control.lifetime import LifetimeResult

    clock = [0.]

    class DevelopmentEvaluator:
        evaluation_transitions = 0
        node_updates = 0

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def evaluate(self, genomes, suite):
            assert self.evaluation_transitions == 0, 'started another generation after the deadline'
            self.evaluation_transitions = 4000
            self.node_updates = 8000
            clock[0] = 11.
            returns = np.zeros((1, len(suite.gains)))
            return LifetimeResult(returns, returns, np.zeros((1, len(suite.gains), 6)),
                                  np.zeros((1, len(suite.gains), 10)))

    monkeypatch.setattr(experiment, 'LifetimeEvaluator', DevelopmentEvaluator)
    monkeypatch.setattr(experiment, 'calibrate_adapter',
                        lambda *args: ObservationAdapter(np.zeros(19), np.ones(19)))
    monkeypatch.setattr(experiment.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(experiment.time, 'monotonic', lambda: clock[0])
    config = experiment.Config(run_dir=str(tmp_path / 'run'), generations=3,
                               development_lives=2, time_limit_seconds=10)
    experiment.train(config)
    saved = json.loads((tmp_path / 'run/latest.json').read_text())
    assert saved['generation'] == 0
    assert saved['stop_reason'] == 'time_limit'
    assert saved['champion']['generation'] == 0

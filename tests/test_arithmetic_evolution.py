import numpy as np
import pytest

from cleanrl.arithmetic_evolution import Config, breed, finite_summary, genome_delta, rank_probabilities, stop_reason
from cleanrl.evolving_programs.genome import Genome


def test_selection_uses_order_not_score_spacing_and_respects_exact_ties():
    scores=np.array([10.,3.,3.,-2.,-np.inf])
    weights=rank_probabilities(scores)
    transformed=rank_probabilities(np.array([1e12,1.,1.,-1e20,-np.inf]))
    np.testing.assert_array_equal(weights,transformed)
    assert weights[1]==weights[2]
    assert weights[0]>weights[1]>weights[3]>weights[4]>0
    assert weights.sum()==pytest.approx(1.)


def test_all_invalid_or_tied_populations_keep_uniform_repair_opportunities():
    np.testing.assert_allclose(rank_probabilities(np.full(8,-np.inf)),np.full(8,1/8))
    np.testing.assert_allclose(rank_probabilities(np.ones(8)),np.full(8,1/8))
    with pytest.raises(ValueError): rank_probabilities([0,np.nan])
    with pytest.raises(ValueError): rank_probabilities([0,np.inf])


def test_failed_lives_cannot_be_hidden_by_safe_zero_or_valid_subset_means():
    result=finite_summary([10000.,0.],[True,False])
    assert result['mean'] is None and result['sem'] is None
    assert result['values']==[10000.,None]
    assert result['valid_lives']==1


def test_breeding_retains_independent_lineages_without_mutating_parents():
    population=[Genome.random(np.random.default_rng(i),19,6,8) for i in range(8)]
    initial=[g.to_json() for g in population]
    config=Config(population=64,initial_nodes=8,max_nodes=32,copy_probability=.25)
    children,births,stats=breed(population,np.arange(8),config,11)
    replay,replayed_births,_=breed(population,np.arange(8),config,11)
    assert [g.to_json() for g in population]==initial
    assert [g.to_json() for g in children]==[g.to_json() for g in replay]
    assert births==replayed_births
    assert stats['distinct_parents']>1
    assert all(0<=b['parent_index']<8 for b in births)
    for child,birth in zip(children,births):
        child.validate(19,6,32)
        assert child is not population[birth['parent_index']]
    children[0].nodes[0].literal += .125
    assert [g.to_json() for g in population]==initial


def test_lineage_delta_reconstructs_a_real_mutated_descendant():
    parent=Genome.random(np.random.default_rng(5),19,6,8)
    child=parent.clone()
    child.mutate(np.random.default_rng(11),19,6,32,2.,1.)
    delta=genome_delta(parent,child)
    previous=parent.to_json()
    replacements={n['node_id']:n for n in delta['nodes']}
    nodes=[replacements.pop(n['node_id'],n) for n in previous['nodes'] if n['node_id'] not in delta['deleted']]
    nodes.extend(replacements.values())
    reconstructed=Genome.from_json({'nodes':nodes,'outputs':delta['outputs'],'next_id':delta['next_id']})
    assert reconstructed.to_json()==child.to_json()


def test_resource_stop_includes_work_after_a_development_boundary():
    config=Config(total_transitions=10000,time_limit_seconds=10,generations=100)
    assert stop_reason(config,5,9000,9) is None
    assert stop_reason(config,5,11000,9)=='transition_budget'
    assert stop_reason(config,5,9000,11)=='time_limit'

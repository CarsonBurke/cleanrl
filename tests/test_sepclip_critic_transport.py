"""CUDA contracts for microscopic critic diagnostics; submit through mlq."""
import torch

from cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7 import ResidualMLP
from scripts.audit_sepclip_critic_transport import feature_stages, spectrum_summary


def test_feature_probe_is_exact_critic_representation():
    torch.manual_seed(1)
    critic=ResidualMLP(17,1,1.).cuda()
    observations=torch.randn(64,17,device='cuda')
    raw,first,final=feature_stages(critic,observations)
    torch.testing.assert_close(raw,critic.first(observations))
    torch.testing.assert_close(first.norm(dim=-1),torch.ones(64,device='cuda'))
    torch.testing.assert_close(critic.readout_gain*critic.head(final),critic(observations),atol=0,rtol=0)


def test_bias_free_first_situ_has_zero_origin_jacobian_and_even_leading_order():
    torch.manual_seed(2)
    critic=ResidualMLP(17,1,1.).cuda()
    origin=torch.zeros(17,device='cuda')
    jacobian=torch.func.jacrev(lambda x:feature_stages(critic,x.unsqueeze(0))[1][0])(origin)
    assert torch.count_nonzero(jacobian)==0
    direction=torch.randn(128,17,device='cuda')
    def odd_fraction(scale):
        pos=critic.first(scale*direction)
        neg=critic.first(-scale*direction)
        return ((pos-neg)*.5).square().sum()/((pos.square().sum()+neg.square().sum())*.5)
    assert odd_fraction(.0001)<odd_fraction(.001)
    assert odd_fraction(.0001)<1e-6


def test_gradient_participation_detects_energy_rank_and_zero_case():
    equal=spectrum_summary(torch.eye(6,device='cuda'))
    torch.testing.assert_close(equal['participation_rank'],torch.tensor(6.,device='cuda'))
    concentrated=torch.zeros(6,6,device='cuda')
    concentrated[0,0]=3
    one=spectrum_summary(concentrated)
    torch.testing.assert_close(one['participation_rank'],torch.tensor(1.,device='cuda'))
    torch.testing.assert_close(one['top1_energy'],torch.tensor(1.,device='cuda'))
    zero=spectrum_summary(torch.zeros(6,6,device='cuda'))
    for value in zero.values():
        assert value==0
        assert torch.isfinite(value)

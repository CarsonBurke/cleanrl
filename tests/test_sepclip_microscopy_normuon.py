"""Warm-history compiled NorMuon instrumentation parity; run through mlq/CUDA."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl import ppo_continuous_action_sepclip_anvil2_v4 as frozen
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as microscopy
from cleanrl.shared.runtime import configure_runtime


def test_best_historical_normuon_compiled_warm_history_parity():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    assert torch.cuda.is_available(), "Submit this contract through mlq on CUDA"
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)),
        single_action_space=gym.spaces.Box(-1., 1., (6,)),
    )
    modules = (frozen, microscopy, microscopy)
    clocks = ("epoch", "rollout", "epoch")
    agents, settings, matrices, auxiliaries, losses = [], [], [], [], []
    for module, clock in zip(modules, clocks):
        kwargs = dict(num_envs=16, num_steps=8, num_minibatches=1,
                      update_epochs=10, compile=True, learning_rate=.0024,
                      normuon_network="actor", matrix_method="normuon",
                      normuon_lr=.005, normuon_momentum=.85,
                      normuon_beta2=.95, matrix_precision="bf16",
                      norm_adv=False, clip_vloss=False, clip_coef_upper=.2)
        if module is microscopy:
            kwargs.update(actor_optimizer="anvil2", slow_clock=clock)
        args = module.validate_args(module.Args(**kwargs))
        args.num_iterations = 2
        torch.manual_seed(1)
        agent = module.Agent(spaces).cuda()
        agent.normalize_matrices()
        matrix, auxiliary = module.make_optimizers(agent, args)

        def learner(obs, native, logp, advantage, returns, values,
                    agent=agent, module=module, args=args):
            return module.ppo_loss(agent, obs, native, logp, advantage,
                                   returns, values, args)

        losses.append(torch.compile(learner, fullgraph=True,
                                   options={"triton.cudagraphs": False}))
        agents.append(agent); settings.append(args)
        matrices.append(matrix); auxiliaries.append(auxiliary)

    def assert_parity():
        reference = agents[0].state_dict()
        for agent in agents[1:]:
            for name, value in agent.state_dict().items():
                torch.testing.assert_close(value, reference[name], atol=0, rtol=0)
        for index, (_, _, reference_state, reference_scale) in enumerate(matrices[0].stacks):
            for matrix in matrices:
                _, _, state, scale = matrix.stacks[index]
                assert scale == reference_scale
                for key in ("velocity", "second"):
                    torch.testing.assert_close(state[key], reference_state[key], atol=0, rtol=0)
                assert torch.count_nonzero(state["velocity"][1]).item() == 0
        # Auxiliary Adam moments must also match, not merely current parameters.
        for candidate in auxiliaries[1:]:
            expected = auxiliaries[0].state_dict()["state"]
            actual = candidate.state_dict()["state"]
            assert actual.keys() == expected.keys()
            for index, fields in expected.items():
                for name, value in fields.items():
                    torch.testing.assert_close(actual[index][name], value, atol=0, rtol=0)

    assert_parity()
    torch.manual_seed(73)
    for rollout in range(2):
        obs = torch.randn(128, 17, device="cuda")
        native = torch.rand(128, 6, device="cuda").clamp(.01, .99)
        with torch.no_grad():
            alpha, beta, values = agents[0].get_policy_and_value(obs)
            values = values.flatten().clone()
            logp = agents[0].action_logprob(alpha, beta, native).clone()
        advantage = torch.randn(128, device="cuda")
        returns = values + advantage
        # Different Anvil clock metadata must have no effect in the NorMuon branch.
        matrices[1].begin_rollout(10)
        matrices[2].begin_rollout(3 + rollout)
        assert_parity()
        for _ in range(10):
            reference_gradients = None
            for agent, fn, matrix, auxiliary in zip(agents, losses, matrices, auxiliaries):
                loss, _ = fn(obs, native, logp, advantage, returns, values)
                agent.zero_grad(set_to_none=True)
                loss.backward()
                for trunk in (agent.actor, agent.critic):
                    torch.nn.utils.clip_grad_norm_(tuple(trunk.parameters()), .5, foreach=True)
                current = {name: p.grad for name, p in agent.named_parameters()}
                if reference_gradients is None:
                    reference_gradients = {name: grad.clone() for name, grad in current.items()}
                else:
                    for name, grad in current.items():
                        torch.testing.assert_close(grad, reference_gradients[name], atol=0, rtol=0)
                matrix.step()
                auxiliary.step()
                agent.normalize_matrices()
            assert_parity()
    for matrix in matrices:
        assert matrix.param_groups[0]["steps"] == 20
        assert any(torch.count_nonzero(state["velocity"][0]).item()
                   for _, _, state, _ in matrix.stacks)
        assert all(torch.count_nonzero(state["second"]).item()
                   for _, _, state, _ in matrix.stacks)

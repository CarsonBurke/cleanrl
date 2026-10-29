"""Write the JAX Disco103 meta-network reference used by tests/test_disco103.py.

Runs google-deepmind/disco_rl itself (not this repo), so it needs that package's
JAX stack in a separate environment, e.g.

    uv venv ~/.cache/disco-rl-jax && uv pip install -p ~/.cache/disco-rl-jax/bin/python \
        "jax[cpu]" dm-haiku rlax distrax chex ml-collections immutabledict dm-env
    PYTHONPATH=../disco_rl ~/.cache/disco-rl-jax/bin/python scripts/disco103_reference.py

Inputs are random in the original [T+1] layout with random taken actions and
episode ends; the lifetime meta-LSTM state is random so its gain path is exercised.
"""

import argparse
from pathlib import Path

import haiku as hk
import jax
import numpy as np

from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils
from disco_rl.update_rules import disco
from disco_rl.value_fns import value_utils

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "tests/fixtures/disco103_reference.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", False)
    rng = np.random.default_rng(args.seed)
    steps, batch, actions, size = 5, 3, 4, 600

    def normal(*shape, scale=1.0):
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    agent_out = dict(logits=normal(steps + 1, batch, actions), y=normal(steps + 1, batch, size, scale=2.0),
                     z=normal(steps + 1, batch, actions, size, scale=2.0))
    target_out = dict(logits=normal(steps + 1, batch, actions), y=normal(steps + 1, batch, size, scale=2.0),
                      z=normal(steps + 1, batch, actions, size, scale=2.0))
    behaviour = dict(logits=normal(steps + 1, batch, actions))
    taken = rng.integers(0, actions, size=(steps + 1, batch)).astype(np.int32)
    is_terminal = np.zeros((steps, batch), dtype=bool)
    is_terminal[1, 0] = is_terminal[3, 2] = is_terminal[4, 1] = True
    extra = dict(
        v_scalar=normal(steps + 1, batch, scale=3.0), adv=normal(steps, batch, scale=2.0),
        normalized_adv=normal(steps, batch), q=normal(steps + 1, batch, actions, scale=3.0),
        qv_adv=normal(steps + 1, batch, actions, scale=2.0),
        normalized_qv_adv=normal(steps + 1, batch, actions), target_out=target_out,
    )
    inputs = types.UpdateRuleInputs(
        observations=np.zeros((steps + 1, batch, 1), np.float32), actions=taken,
        rewards=normal(steps, batch, scale=2.0), is_terminal=is_terminal, agent_out=agent_out,
        behaviour_agent_out=behaviour, extra_from_rule=extra,
    )

    settings = agent_lib.get_settings_disco()
    rule = disco.DiscoUpdateRule(**settings.update_rule)
    with np.load(disco.__file__.replace("disco.py", "weights/disco_103.npz")) as archive:
        flat = {key: archive[key] for key in archive.files}
    params = {}
    for key in flat:
        module = key.rsplit("/", 1)[0]
        params[module] = dict(w=flat[f"{module}/w"], b=flat[f"{module}/b"])
    _, state = rule.init_params(jax.random.PRNGKey(0))
    lifetime = hk.LSTMState(hidden=normal(128, scale=0.5), cell=normal(128, scale=0.5))
    state = jax.tree.map(lambda x: x, state)
    (module,) = state.keys()
    state[module]["meta_rnn_state"] = lifetime
    meta_out, new_state = rule._eta_apply(params, state, None, inputs, axis_name=None)
    new_lifetime = new_state[module]["meta_rnn_state"]

    # Categorical value transforms and EMA normalization (value_utils, utils.MovingAverage).
    value_targets = np.concatenate((normal(64, scale=50.0), np.float32([0.0, 1.0, -7.0, 1e5, -1e5])))
    value_logits = normal(value_targets.size, 601, scale=2.0)
    value_ce = value_utils.value_loss_from_target(
        value_logits, value_targets, nonlinear_transform=True, categorical_value=True,
        max_abs_value=300.0)
    value_scalars = value_utils.get_values_from_net_outs(
        value_logits[None], categorical_value=True, max_abs_value=300.0,
        nonlinear_transform=True)[0]
    moving = utils.MovingAverage(np.zeros((), np.float32), decay=0.99, eps=1e-6)
    ema_state = moving.init_state()
    ema_samples = normal(3, 40, scale=2.0) + 1.5
    for sample in ema_samples:
        ema_state = moving.update_state(sample, ema_state, None)
    ema_normalized = moving.normalize(ema_samples[-1], ema_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        value_targets=value_targets, value_logits=value_logits, value_ce=np.asarray(value_ce),
        value_scalars=np.asarray(value_scalars), ema_samples=ema_samples,
        ema_normalized=np.asarray(ema_normalized),
        logits=agent_out["logits"], y=agent_out["y"], z=agent_out["z"],
        target_logits=target_out["logits"], target_y=target_out["y"], target_z=target_out["z"],
        behaviour_logits=behaviour["logits"], actions=taken, rewards=inputs.rewards,
        is_terminal=is_terminal.astype(np.float32),
        **{key: value for key, value in extra.items() if key != "target_out"},
        lifetime_state=np.stack((lifetime.hidden, lifetime.cell)),
        pi_hat=np.asarray(meta_out["pi"]), y_hat=np.asarray(meta_out["y"]),
        z_hat=np.asarray(meta_out["z"]),
        new_lifetime_state=np.stack((np.asarray(new_lifetime.hidden), np.asarray(new_lifetime.cell))),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

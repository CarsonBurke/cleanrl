"""Disco103, the discovered RL update rule of Oh et al. (Nature 2025), in PyTorch.

Frozen port of google-deepmind/disco_rl (Apache 2.0): the meta-network
(``networks/meta_nets.py``), its input transforms (``update_rules/disco.py``
``get_input_option`` and ``input_transforms.py``), the EMA normalizer
(``utils.MovingAverage``) and the rlax value transforms its categorical value
function uses. ``disco103_weights.npz`` is the released ``disco_103.npz``,
unchanged; ``tests/test_disco103.py`` pins this port to the JAX original.

The meta-network is permutation-equivariant over a discrete action axis: per
action it embeds (pi, mu, target pi, z, target z, Q, Q-V, normalized Q-V,
taken?) with mean-pooled 1x1 convolutions, so it accepts any finite action set
of size K. Callers present that set on the last axis with the TAKEN ACTION AT
INDEX 0 (``select_a`` and the one-hot channel read index 0).

Time layout. The original consumes [T+1] agent outputs and takes "td pairs"
(x[:-1], x[1:]). Here every per-state input is split into a current [T] and a
next [T] tensor instead, so a caller can evaluate the successor state of a
completed episode on its final observation; with ``next_x = x[1:]`` the two
layouts are identical. ``resets[t]`` marks an episode boundary after step t
(the backward per-trajectory LSTM restarts there); ``terminations[t]`` is the
discount input, so truncations reset without zeroing the discount.

Shapes: T time, B batch, K actions, P = PREDICTION_SIZE.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WEIGHTS = Path(__file__).with_name("disco103_weights.npz")
PREDICTION_SIZE = 600
TRAJECTORY_HIDDEN = 256
LIFETIME_HIDDEN = 128

# Disco103 agent-loss coefficients (agent.get_settings_disco().hyper_params).
PI_COST = 1.0
Y_COST = 1.0
Z_COST = 1.0
AUX_POLICY_COST = 1.0
VALUE_COST = 0.2
TARGET_PARAMS_COEFF = 0.9
TD_LAMBDA = 0.95


class MetaInputs(NamedTuple):
    logits: torch.Tensor                    # [T,B,K] online policy at s_t
    next_logits: torch.Tensor               # [T,B,K] online policy at s_t+1 (pi-weighted z)
    behaviour_logits: torch.Tensor          # [T,B,K]
    target_logits: torch.Tensor             # [T,B,K] target-network policy at s_t
    rewards: torch.Tensor                   # [T,B]
    terminations: torch.Tensor              # [T,B] 1 where s_t+1 is terminal
    resets: torch.Tensor                    # [T,B] 1 where the episode ends after step t
    values: torch.Tensor                    # [T,B] online V(s_t)
    next_values: torch.Tensor               # [T,B] online V(s_t+1)
    advantages: torch.Tensor                # [T,B] Retrace Q target minus target V
    normalized_advantages: torch.Tensor     # [T,B]
    y: torch.Tensor                         # [T,B,P] online y(s_t) logits
    next_y: torch.Tensor                    # [T,B,P]
    target_y: torch.Tensor                  # [T,B,P]
    next_target_y: torch.Tensor             # [T,B,P]
    z: torch.Tensor                         # [T,B,K,P] online z(s_t, a) logits
    next_z: torch.Tensor                    # [T,B,K,P]
    target_z: torch.Tensor                  # [T,B,K,P]
    next_target_z: torch.Tensor             # [T,B,K,P]
    target_q: torch.Tensor                  # [T,B,K] target Q(s_t, a)
    qv_advantages: torch.Tensor             # [T,B,K] target Q(s_t, a) - target V(s_t)
    normalized_qv_advantages: torch.Tensor  # [T,B,K]


def signed_log1p(x):
    """rlax.signed_logp1."""
    return torch.sign(x) * torch.log1p(x.abs())


def signed_hyperbolic(x, eps=1e-3):
    """rlax.signed_hyperbolic, the value compression h(x)."""
    return torch.sign(x) * (torch.sqrt(x.abs() + 1.0) - 1.0) + eps * x


def signed_parabolic(x, eps=1e-3):
    """rlax.signed_parabolic, the exact inverse of ``signed_hyperbolic``."""
    z = torch.sqrt(1.0 + 4.0 * eps * (eps + 1.0 + x.abs())) / (2.0 * eps) - 1.0 / (2.0 * eps)
    return torch.sign(x) * (z.square() - 1.0)


class TwoHotValue(nn.Module):
    """Disco's categorical value: two-hot bins over h(x) on [-max_abs, max_abs]."""

    def __init__(self, num_bins, max_abs_value):
        super().__init__()
        self.num_bins = int(num_bins)
        self.max_abs_value = float(max_abs_value)
        self.register_buffer("support", torch.linspace(-max_abs_value, max_abs_value, num_bins))

    def scalar(self, logits):
        """rlax.transform_from_2hot of softmax(logits), mapped back through h^-1."""
        return signed_parabolic(torch.softmax(logits, -1) @ self.support)

    def loss(self, logits, targets):
        """Cross-entropy to rlax.transform_to_2hot(h(targets)) without the dense target."""
        scaled = (signed_hyperbolic(targets).clamp(-self.max_abs_value, self.max_abs_value)
                  + self.max_abs_value) * ((self.num_bins - 1) / (2.0 * self.max_abs_value))
        lower = scaled.floor().clamp_max(self.num_bins - 1)
        upper_weight = scaled - lower
        log_probs = torch.log_softmax(logits, -1)
        index = lower.long().unsqueeze(-1)
        lower_log_prob = log_probs.gather(-1, index).squeeze(-1)
        upper_log_prob = log_probs.gather(-1, (index + 1).clamp_max(self.num_bins - 1)).squeeze(-1)
        return -((1.0 - upper_weight) * lower_log_prob + upper_weight * upper_log_prob)


def ema_init(device=None):
    """utils.MovingAverage state: (first moment, second moment, decay product)."""
    return torch.tensor([0.0, 0.0, 1.0], device=device)


def ema_update(state, values, decay=0.99):
    mean = values.mean()
    first = decay * state[0] + (1.0 - decay) * mean
    second = decay * state[1] + (1.0 - decay) * values.square().mean()
    return torch.stack((first, second, state[2] * decay))


def ema_normalize(state, values, subtract_mean=True, eps=1e-6, root_eps=1e-12):
    debias = 1.0 / (1.0 - state[2])
    mean = state[0] * debias
    variance = (state[1] * debias - mean.square()).clamp_min(0.0)
    centred = values - mean if subtract_mean else values
    return centred / (torch.sqrt(variance + root_eps) + eps)


def categorical_kl(p_logits, q_logits):
    """rlax.categorical_kl_divergence: KL(softmax(p) || softmax(q)) over the last axis."""
    p_log = torch.log_softmax(p_logits, -1)
    return (p_log.exp() * (p_log - torch.log_softmax(q_logits, -1))).sum(-1)


def _mlp(x, layers):
    """hk.nets.MLP: ReLU between layers, none after the last."""
    for index, (weight, bias) in enumerate(layers):
        x = F.linear(x, weight, bias)
        if index < len(layers) - 1:
            x = F.relu(x)
    return x


def _conv_blocks(x, layers):
    """meta_nets._conv1d_net: per-action 1x1 conv over [x, mean over actions], ReLU."""
    for weight, bias in layers:
        x = torch.cat((x, x.mean(-2, keepdim=True).expand_as(x)), -1)
        x = F.relu(F.linear(x, weight, bias))
    return x


def lstm_gates(gates, cell):
    """hk.LSTM given precomputed gates; the forget gate carries Sonnet's +1 bias."""
    i, g, f, o = gates.chunk(4, -1)
    cell = torch.sigmoid(f + 1.0) * cell + torch.sigmoid(i) * torch.tanh(g)
    return torch.sigmoid(o) * torch.tanh(cell), cell


class Disco103(nn.Module):
    """The frozen Disco103 meta-network: rollout features -> (pi_hat, y_hat, z_hat)."""

    def __init__(self, weights=WEIGHTS):
        super().__init__()
        with np.load(weights) as archive:
            raw = {key: torch.as_tensor(archive[key], dtype=torch.float32) for key in archive.files}
        self._names = []

        def linear(prefix):
            # Haiku stores [in, out] (Conv1D: [1, in, out]); F.linear wants [out, in].
            weight = raw[f"{prefix}/w"]
            weight = weight.reshape(-1, weight.shape[-1]).T.contiguous()
            name = prefix.replace("/~/", "/").replace("/", "__").replace("~", "")
            self.register_buffer(f"{name}__w", weight)
            self.register_buffer(f"{name}__b", raw[f"{prefix}/b"])
            self._names.append(name)
            return name

        def embedding_nets(prefix):
            return dict(
                y_net=(linear(f"{prefix}mlp/~/linear_0"), linear(f"{prefix}mlp/~/linear_1")),
                z_net=(linear(f"{prefix}mlp_1/~/linear_0"), linear(f"{prefix}mlp_1/~/linear_1")),
                policy_net=(linear(f"{prefix}sequential/conv1_d"),
                            linear(f"{prefix}sequential/conv1_d_1")),
            )

        self._trajectory = embedding_nets("lstm/")
        self._trajectory_rnn = linear("lstm/lstm/linear")
        self._lifetime_gain = linear("lstm/linear")
        self._meta_input = linear("lstm/linear_1")
        self._y_head = linear("lstm/linear_2")
        self._z_head = linear("lstm/linear_3")
        self._policy_conv = linear("lstm/sequential_1/conv1_d")
        self._policy_head = linear("lstm/linear_4")
        lifetime = "lstm/~/meta_lstm/~unroll/"
        self._lifetime = embedding_nets(lifetime)
        self._lifetime_embed = linear(f"{lifetime}mlp_2/~/linear_0")
        self._lifetime_rnn = linear(f"{lifetime}lstm/linear")
        if 2 * len(set(self._names)) != len(raw):
            raise ValueError(f"Disco103 weights: consumed {2 * len(self._names)} of {len(raw)} tensors")

    def _layer(self, name):
        return getattr(self, f"{name}__w"), getattr(self, f"{name}__b")

    def initial_state(self, device=None):
        """Lifetime meta-LSTM state (hidden, cell); zeros, as hk.LSTM.initial_state."""
        return torch.zeros(2, LIFETIME_HIDDEN, device=device)

    def _features(self, f: MetaInputs, nets):
        """meta_nets._construct_input with disco.get_input_option(): [T,B,27] and [T,B,K,2]."""
        y_net = [self._layer(name) for name in nets["y_net"]]
        z_net = [self._layer(name) for name in nets["z_net"]]
        policy_net = [self._layer(name) for name in nets["policy_net"]]
        pi = torch.softmax(f.logits, -1)
        next_pi = torch.softmax(f.next_logits, -1)
        mu = torch.softmax(f.behaviour_logits, -1)
        target_pi = torch.softmax(f.target_logits, -1)

        def embed_y(logits):
            return _mlp(torch.softmax(logits, -1), y_net).squeeze(-1)

        def embed_z(logits):
            return _mlp(torch.softmax(logits, -1), z_net).squeeze(-1)

        z, next_z = embed_z(f.z), embed_z(f.next_z)
        target_z, next_target_z = embed_z(f.target_z), embed_z(f.next_target_z)
        action_conditional = torch.stack((
            pi, mu, target_pi, z, target_z,
            signed_log1p(f.target_q), signed_log1p(f.qv_advantages), f.normalized_qv_advantages,
            F.one_hot(torch.zeros_like(f.target_q[..., 0], dtype=torch.long),
                      f.target_q.shape[-1]).to(pi.dtype),
        ), -1)
        action_embedding = _conv_blocks(action_conditional, policy_net)
        scalars = torch.stack((
            pi[..., 0], mu[..., 0], signed_log1p(f.rewards), 1.0 - f.terminations,
            signed_log1p(f.values), signed_log1p(f.next_values),
            signed_log1p(f.advantages), f.normalized_advantages, target_pi[..., 0],
            embed_y(f.y), embed_y(f.next_y), embed_y(f.target_y), embed_y(f.next_target_y),
            z[..., 0], (pi * z).sum(-1), (next_pi * next_z).sum(-1),
            z.amax(-1), next_z.amax(-1),
            target_z[..., 0], (pi * target_z).sum(-1), (next_pi * next_target_z).sum(-1),
            target_z.amax(-1), next_target_z.amax(-1),
        ), -1)
        features = torch.cat((scalars, action_embedding.mean(-2), action_embedding[..., 0, :]), -1)
        return features, action_embedding

    def forward(self, f: MetaInputs, state):
        """Return (pi_hat [T,B,K], y_hat [T,B,P], z_hat [T,B,P], next lifetime state [2,H])."""
        features, action_embedding = self._features(f, self._trajectory)
        steps, batch = features.shape[:2]

        # Per-trajectory LSTM, unrolled backwards and restarted at episode ends.
        weight, bias = self._layer(self._trajectory_rnn)
        input_weight, hidden_weight = weight.split((features.shape[-1], TRAJECTORY_HIDDEN), 1)
        input_gates = F.linear(features, input_weight, bias)
        keep = (1.0 - f.resets).unsqueeze(-1)
        hidden = features.new_zeros(batch, TRAJECTORY_HIDDEN)
        cell = torch.zeros_like(hidden)
        outputs = [None] * steps
        for t in reversed(range(steps)):
            hidden, cell = hidden * keep[t], cell * keep[t]
            hidden, cell = lstm_gates(input_gates[t] + F.linear(hidden, hidden_weight), cell)
            outputs[t] = hidden
        x = torch.stack(outputs)

        # Multiplicative conditioning on the lifetime meta-LSTM's hidden state.
        x = x * F.linear(state[0], *self._layer(self._lifetime_gain))
        meta_input_embedding = F.linear(x, *self._layer(self._meta_input))
        y_hat = F.linear(x, *self._layer(self._y_head))
        z_hat = F.linear(x, *self._layer(self._z_head))
        per_action = torch.cat((x.unsqueeze(-2).expand(*x.shape[:2], action_embedding.shape[-2], -1),
                                action_embedding), -1)
        per_action = _conv_blocks(per_action, [self._layer(self._policy_conv)])
        pi_hat = F.linear(per_action, *self._layer(self._policy_head)).squeeze(-1)

        # Lifetime meta-LSTM: one step on the batch-time mean of this update's embedding.
        lifetime_features, _ = self._features(f, self._lifetime)
        lifetime_y = _mlp(torch.softmax(y_hat, -1),
                          [self._layer(name) for name in self._lifetime["y_net"]])
        embedding = F.linear(torch.cat((lifetime_features, meta_input_embedding, lifetime_y), -1),
                             *self._layer(self._lifetime_embed))
        weight, bias = self._layer(self._lifetime_rnn)
        gates = F.linear(torch.cat((embedding.mean((0, 1)), state[0])), weight, bias)
        new_state = torch.stack(lstm_gates(gates, state[1]))
        return pi_hat, y_hat, z_hat, new_state

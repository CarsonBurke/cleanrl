"""
Dreamer4 World Model for MuJoCo Continuous Control
====================================================
Self-contained implementation of the Dreamer4 architecture adapted for
state-based continuous control tasks (HalfCheetah, Hopper, Walker2d).

Key components:
- StateTokenizer: encodes observation vectors into latent tokens
- DynamicsWorldModel: shortcut flow matching world model with PPO/PMPO
- AxialSpaceTimeTransformer: backbone with spatial/temporal attention + GRU
- ContinuousActionEmbedder: action embedding/unembedding for policy
"""

from .utils import (
    Experience,
    Actions,
    StateTokenizer,
    SymExpTwoHot,
    calc_gae,
)

from .transformer import AxialSpaceTimeTransformer

from .actions import ContinuousActionEmbedder

from .world_model import DynamicsWorldModel

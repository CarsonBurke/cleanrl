"""Evolutionary collective control v2.

No PPO or SGD: fixed recurrent graph residents are mutated and accepted only when
paired replacement rollouts improve raw Gymnasium team return. v2 batches all
rollout episodes through CleanRL's native threaded MuJoCo vector environment and
logs standard episodic-return TensorBoard scalars for direct benchmark comparison.
"""

from cleanrl.collective_control.evolve import main


if __name__ == "__main__":
    main()

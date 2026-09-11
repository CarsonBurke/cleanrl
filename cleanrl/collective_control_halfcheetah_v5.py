"""Evolutionary collective control v5.

No PPO or SGD. v5 keeps comparative slot-replacement selection, batches proposal
and frozen-incumbent confirmation teams into one CUDA/native-MuJoCo sweep, and
uses the live raw Gymnasium episodic-return contract for comparison.
"""

from cleanrl.collective_control.evolve import main


if __name__ == "__main__":
    main()

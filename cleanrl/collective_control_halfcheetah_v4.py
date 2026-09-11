"""Evolutionary collective control v4.

No PPO or SGD. Proposal teams are batched into one CUDA/native MuJoCo
evaluation; confirmation remains sequential by victim so accepted replacements
can co-adapt within a generation.
"""

from cleanrl.collective_control.evolve import main


if __name__ == "__main__":
    main()

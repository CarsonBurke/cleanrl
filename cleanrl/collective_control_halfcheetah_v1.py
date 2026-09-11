"""Evolutionary collective control v1.

No PPO or SGD: fixed recurrent graph residents are mutated and accepted only when
paired replacement rollouts improve team return. Standalone fitness is diagnostic;
comparative slot contribution is the selection signal.
"""

from cleanrl.collective_control.evolve import main


if __name__ == "__main__":
    main()

"""Evolutionary collective control v3.

No PPO or SGD. v3 keeps paired team-level replacement selection and native
vectorized MuJoCo rollouts, while logging raw fixed-horizon team returns to the
standard ``charts/episodic_return`` tag every generation and separately logging
frozen development returns.
"""

from cleanrl.collective_control.evolve import main


if __name__ == "__main__":
    main()

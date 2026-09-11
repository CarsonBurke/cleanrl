"""Evolutionary collective control v6.

No PPO or SGD. v6 uses short proposal rollouts, full-horizon confirmation only
for shortlists, and one full-horizon development score when due. It retains the
team-only paired replacement objective while making the transition budget explicit.
"""

from cleanrl.collective_control.evolve import main


if __name__ == "__main__":
    main()

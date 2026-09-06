import argparse
from pathlib import Path
from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: Callable,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    try:
        agent = Model(envs).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()

        obs, _ = envs.reset()
        episodic_returns = []
        while len(episodic_returns) < eval_episodes:
            with torch.no_grad():
                actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if not info or "episode" not in info:
                        continue
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]
            obs = next_obs

        return episodic_returns
    finally:
        envs.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a local bounded-action Beta PPO checkpoint. Run CUDA evaluation through mlq.",
        epilog="Legacy Gaussian checkpoints require their original Agent class via evaluate(Model=...).",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="local Beta PPO .cleanrl_model file")
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--run-name", default="ppo_beta_eval")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    if not args.model_path.is_file():
        parser.error("--model-path must name an existing local checkpoint file")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this evaluation through mlq")
    from cleanrl.ppo_continuous_action import Agent, make_env

    return evaluate(
        str(args.model_path),
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=args.run_name,
        Model=Agent,
        device=torch.device("cuda"),
        capture_video=args.capture_video,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()

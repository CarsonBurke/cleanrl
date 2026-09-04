# Resume Training


A common question we get asked is how to set up model checkpoints to continue training. In this document, we take this [PPO example](https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo_gridnet.py) to explain that question.

## Save model checkpoints

Checkpoint only after a completed update. The example below uses a monotonic ten-minute cadence, atomically replaces one rolling local artifact, and writes a final state only when that update has not already been committed.

```python linenums="1"
import random
import time
from pathlib import Path

import numpy as np
import torch

from cleanrl_utils.checkpointing import atomic_save

num_updates = args.total_timesteps // args.batch_size
checkpoint_path = Path(wandb.run.dir if args.track else "checkpoints") / "agent.pt"
checkpoint_interval_seconds = 600
next_checkpoint_at = time.monotonic() + checkpoint_interval_seconds
last_checkpointed_update = None

def save_checkpoint(update):
    payload = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "global_step": global_step,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_save(checkpoint_path, lambda staging_path: torch.save(payload, staging_path))
    if args.track:
        wandb.save(str(checkpoint_path), policy="now")

for update in range(1, num_updates + 1):
    # ... do rollouts and train models

    # This boundary is safe: the update and optimizer step are complete.
    if time.monotonic() >= next_checkpoint_at:
        save_checkpoint(update)
        last_checkpointed_update = update
        next_checkpoint_at = time.monotonic() + checkpoint_interval_seconds

if last_checkpointed_update != num_updates:
    save_checkpoint(num_updates)
```

Then we could run the following to train our agents

```
python ppo_gridnet.py --prod-mode --capture_video
```

If the training was terminated early, we can still see the last updated model `agent.pt` in W&B like in this URL [https://wandb.ai/costa-huang/cleanRL/runs/21421tda/files](https://wandb.ai/costa-huang/cleanRL/runs/21421tda/files) or as follows

<iframe src="https://wandb.ai/costa-huang/cleanRL/runs/21421tda/files" style="width:100%; height:500px" title="CleanRL CartPole-v1 Example"></iframe>


## Resume training

The checkpoint contains the model, optimizer, progress counters, and process RNG state rather than model weights alone. Download and restore it before continuing:

This is a parameter-and-process resume recipe, not a bit-exact simulator resume.
Most Gym environments and wrappers do not expose their live state, current
observation, or action-space RNG for serialization, so a reconstructed
environment starts a different rollout. Exact continuation requires an
environment that explicitly supports saving and restoring all of that state at
the same update boundary.


```python linenums="1"
num_updates = args.total_timesteps // args.batch_size
starting_update = 1
last_checkpointed_update = None

if args.track and wandb.run.resumed:
    api = wandb.Api()
    run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
    model = run.file("agent.pt")
    model.download(f"models/{experiment_name}/", replace=True)
    checkpoint = torch.load(
        f"models/{experiment_name}/agent.pt",
        map_location=device,
        weights_only=False,
    )
    agent.load_state_dict(checkpoint["agent"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if checkpoint["torch_cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    global_step = checkpoint["global_step"]
    starting_update = checkpoint["update"] + 1
    last_checkpointed_update = checkpoint["update"]
    print(f"resumed after update {checkpoint['update']}")

next_checkpoint_at = time.monotonic() + checkpoint_interval_seconds
for update in range(starting_update, num_updates + 1):
    # ... do rollouts and train models

    if time.monotonic() >= next_checkpoint_at:
        save_checkpoint(update)
        last_checkpointed_update = update
        next_checkpoint_at = time.monotonic() + checkpoint_interval_seconds

if last_checkpointed_update != num_updates:
    save_checkpoint(num_updates)
```

To resume training, note the ID of the experiment is `21421tda` as in the URL [https://wandb.ai/costa-huang/cleanRL/runs/21421tda](https://wandb.ai/costa-huang/cleanRL/runs/21421tda), so we need to pass in the ID via environment variable to trigger the resume mode of W&B:

```
WANDB_RUN_ID=21421tda WANDB_RESUME=must python ppo_gridnet.py --prod-mode --capture_video
``` 
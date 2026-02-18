
import sys
from tensorboard.backend.event_processing import event_accumulator
import os

def get_last_return(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    try:
        returns = ea.scalars.Items('charts/episodic_return')
        if returns:
            # find closest to 1M
            closest_1M = min(returns, key=lambda x: abs(x.step - 1000000))
            print(f"Return at ~1M ({closest_1M.step}): {closest_1M.value}")
            return returns[-1].value, returns[-1].step
    except Exception as e:
        print(f"Error: {e}")
        pass
    return None, None

if __name__ == "__main__":
    run_dir = sys.argv[1]
    event_files = [f for f in os.listdir(run_dir) if "events.out.tfevents" in f]
    if not event_files:
        print("No event file found")
        sys.exit(0)
    event_file = event_files[0]
    val, step = get_last_return(os.path.join(run_dir, event_file))
    if val:
        print(f"Last Return: {val} at step {step}")

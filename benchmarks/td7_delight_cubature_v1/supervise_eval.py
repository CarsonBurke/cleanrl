"""Use greedy evaluation to cull DG runs whose exploration scales may differ."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import autocull

# Process-local selection: reuse the existing queue-aware supervisor and its
# conservative thresholds. Training return remains logged as training return.
autocull.RETURN_TAG = 'eval/episodic_return'
autocull.CurveCache.TAGS = frozenset({autocull.RETURN_TAG, autocull.VALUE_LOSS_TAG})

if __name__ == '__main__':
    autocull.main()

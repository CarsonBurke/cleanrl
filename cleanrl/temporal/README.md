# Temporal Context Aggregation

Lightweight transformer backbones that process short observation histories to give the policy temporal context. Simpler than the full STSTS axial architecture -- these use a single temporal transformer with CLS token readouts.

## Variants

| File | Architecture |
|-|-|
| `_temporal_separate_cls` | Separate CLS tokens per head, separate transformer layers |
| `_temporal_shared_3cls` | Shared transformer + 3 CLS tokens (actor, critic, SDE) |
| `_temporal_shared_3cls_ctx1` | Same with context length = 1 (ablation) |
| `_temporal_shared_last` | Shared transformer, use last hidden state instead of CLS |

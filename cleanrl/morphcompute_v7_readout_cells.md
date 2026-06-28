# MorphCompute v7 Note: Paid Readout Cells

## Goal
Replace global active-weighted pooling with dedicated readout cells. The next version should have exactly `action_dim` actor readout cells in the substrate, so each action dimension has a specific cell that projects directly into the actor head. The critic should use either one value readout cell or its own small fixed set of critic readout cells.

## Intended Change
- Add readout cells to the substrate rather than pooling all cells into one vector.
- Actor substrate should expose `action_dim` readout cell states.
- Actor Beta heads should be direct per-action projections from those readout cells:
  - `alpha_i = 1 + softplus(alpha_head(readout_cell_i))`
  - `beta_i = 1 + softplus(beta_head(readout_cell_i))`
- Critic should avoid unconstrained global pooling too. Prefer one learned value readout cell that projects to scalar value.
- Readout cells must receive information through the same paid sparse edge mechanism as ordinary cells, or through a separately logged paid readout-edge budget.

## Why
The current active-weighted pooling can bypass communication. Independent cells can each see the same observation embedding and the final pooling layer aggregates them for free. That makes sparse internal edges easy to prune. Dedicated readout cells force useful information to move through explicit substrate paths before reaching the policy/value heads.

## Acceptance Criteria
- No global active-weighted mean pooling for actor output.
- Actor has exactly `action_dim` readout cells.
- Actor/critic readout connectivity is included in compute accounting.
- Log readout edge counts/fractions separately from internal substrate edges.
- Keep PPO replay in native Beta `z` unchanged.

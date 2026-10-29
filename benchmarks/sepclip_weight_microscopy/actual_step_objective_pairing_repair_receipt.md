# Fixed actual motions across dense objectives

Authorized additive diagnostic repair only. No frozen trainer, existing readiness report, queue job or source submission receipt was modified. Both modified files pass AST parsing; no tests, models or jobs were executed by this reviewer.

| File | SHA256 before | SHA256 after |
| --- | --- | --- |
| scripts/audit_sepclip_actual_oracle_step.py | 80e4dbc117808da90e9b5b87febc856a2c26d708ac3a4e9c2a56542ffa64c8e5 | 0d23d4b61059a255b7c566fd8088ebea629f357905a3fd7b35928cbb5b848953 |
| tests/test_sepclip_actual_oracle_step.py | c8d905cb782dabbe57186d4cd6e3760a16bbbe0f9c6198227af00a4f67079f17 | e6a5ed7b073bbbdd8c017b241016f4ed7b4c48430c6f69c30d7b48b45638c05f |

Existing finite_normalized cases, fits, ranks and keys remain unchanged. Each horizon additionally emits fixed_actual_motion_objective_pairings for finite_raw, finite_normalized and bootstrap_normalized. Every objective uses the same first-update and endpoint total JVP, tangent-only JVP and actual finite logit displacement. No objective-specific direction fitting, additional actor forward, Jacobian or SVD is introduced.

For each objective/motion, outputs retain signed per-replica pairings, their arithmetic mean, mean-field pairing, conditional Monte Carlo SE and approximate Student-t95 interval. The pairing divides by the number of ALL selected states, summing logit/action contributions. A finite logit displacement paired with a local covector is explicitly not a finite return or an all-time policy derivative.

The separate paired_estimated_tail_increment block subtracts finite_normalized from bootstrap_normalized gains at the SAME replica index, then estimates SE and interval from those differences. It describes the added estimated terminal critic reward, not true critic bias. Index alignment is assumed from the shared dense producer; absent replica IDs, independent field reordering cannot be detected. An additional correlated-replica test has tail-difference SE0.5 versus SE14.5 if incorrectly treating the objective estimates as independent.

Inputs require explicit first_action_only=True, declared integer replica count>=2, finite matching state/logit layouts, all three objective means and replicas, mean/replica consistency, and horizon/discount/tail-weight metadata. Historical selected-state mapping and snapshot checks remain in place. Conditional intervals address noise replicas only; they do not resolve selected-state or seed uncertainty and are unstable with two replicas. The objectives remain separate, including the estimated terminal critic tail in bootstrap_normalized.

New CUDA contract tests use independently specified positive per-state raw/normalized scale factors to produce opposite pairing signs; a terminal-tail field independently reverses the normalized sign. They check all six fixed motion variants, signed replicas, the state denominator, conditional SE and rejection of inconsistent means, nonfinite fields, layout/count mismatches, missing objectives, wrong tail metadata and incorrect first-action scope.

Execution remains gated on independent review and queued CUDA contracts. Root must reconcile the amended source/test hashes against queued attempt-zero tests11770/11776 and dependent source-pinned11782 before execution. This receipt does not claim those jobs were updated or the tests passed.

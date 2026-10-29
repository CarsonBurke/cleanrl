# Independent static review: objective pairings and finite guards

Result: approved for queued numerical validation. This is a source-only review, not a passing test result or new physical measurement. No models, tests or jobs were executed or submitted by this reviewer; no trainer or existing diagnostic source was edited by this reviewer.

Reviewed final byte hashes:

| File | SHA256 |
| --- | --- |
| scripts/audit_sepclip_actual_oracle_step.py | 0d23d4b61059a255b7c566fd8088ebea629f357905a3fd7b35928cbb5b848953 |
| tests/test_sepclip_actual_oracle_step.py | e6a5ed7b073bbbdd8c017b241016f4ed7b4c48430c6f69c30d7b48b45638c05f |
| scripts/audit_sepclip_dense_pg.py | e7c6fd46e185e158bc7b00b11b9f86f94d3dbd885d6bc6bfea80146f57f2ddac |
| tests/test_sepclip_dense_pg.py | c5d61ba145df32a0e9b1fc2e6a86345ae3be87f820607874af73b8e03d987d98 |
| scripts/audit_sepclip_closed_loop_update.py | 761e5ffd86d7432cdc47c21c9ad3a77a8c9d0f50ea39c40b608defdd812cef65 |
| tests/test_sepclip_closed_loop_update.py | 58f02e6bb956b0f08c1a385838e29fb8a4b7a6012352901cf039a1e62c166808 |

Required existing validation gates: 11764, 11770, 11776 and 11795. This review neither changes their state nor claims their execution or success. Any future execution must use mlq with maxParallelRuns1 under the supervisor's evidence gates.

The additive fixed-motion helper sums the signed logit-coordinate pairing and divides by the number of selected states. It uses the same actual first/end total JVP, tangent JVP and finite logit delta for each raw, normalized and bootstrap-normalized field, with no refitting. Finite means and replicas must match layouts, declared replica count and replica means. Explicit first-action scope and consistent horizon/discount/tail metadata are required. Existing normalized fits and geometry fields retain their schema and computation.

The estimated tail increment subtracts bootstrap-normalized minus finite-normalized gains within each identical replica index, then computes its own standard error and Student-t interval. The correlated fixture's gains [-3.5,-2.5] give mean -3 and SE .5; independent subtraction of objective errors would incorrectly give about 14.5. The intervals describe approximate conditional Monte Carlo uncertainty for fixed states and motions; they do not include state or training-seed uncertainty, and two-replica intervals are unstable. Replica alignment assumes the common producer order, because missing replica IDs cannot identify independent reordering. Added critic-tail credit is not true critic bias, and finite logit pairing is not a finite-return measurement or an all-time policy derivative.

The final finite guards reject invalid policy/draw/physical-transition inputs, required score/JVP/baseline/terminal fields, normalization before clipping and trajectory return accumulations. Saved selected physical actions, raw rewards and next observations are checked before replay; replay error scalars are also checked before threshold comparisons. Historical GAE inputs, critic replay and error scalars are checked, and reward scales must be positive. Observation variances must be nonnegative, and observation normalization cannot hide overflow behind clipping. Historical reward replay now checks raw/std before the identical clip/cast, closing the analogous finite-input overflow hole. Strict allow_nan=False output serialization additionally prevents successful JSON containing nonfinite downstream statistics.

The focused saved-input, historical-credit and normalization fixtures exercise concrete earlier bypasses: NaN comparisons previously failed to reject; clipping could hide an infinite preclip quotient. The historical overflow fixture uses finite float64 captured raw reward and scale, while preserving the historical GAE fields, so it targets the quotient guard rather than an earlier malformed-input guard. Invalid closed-loop policy fails before a draw or environment step. These fixtures have been read, not executed.

Valid numerical laws are preserved: dense normalized reward accumulation clips FP64 values; closed-loop normalization clips then casts each reward to FP32 before FP64 accumulation. The repairs do not establish equivalence of those laws. Source approval does not establish PG replicate reliability, nonlinear JVP numerical correctness, sampler-law score validity, useful actor reachability, causal optimizer superiority or a selected chart candidate. Those conclusions remain dependent on independent queued numerical and physical evidence.

One nonblocking implementation detail remains: paired scalar tail statistics reconstruct tiny CPU tensors from serialized gains, whereas the model and primary pairings use CUDA. This is scalar postprocessing, not a CPU model evaluation; retaining original CUDA gains would avoid that detour.

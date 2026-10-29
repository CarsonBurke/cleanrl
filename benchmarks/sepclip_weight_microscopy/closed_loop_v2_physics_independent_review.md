# Independent physics review: V2 JVP ownership repair

Static preparation APPROVED. No remaining blocker found in the proposed V2 change. This review did not execute models, tests, or jobs and does not establish that the Torch failure is repaired numerically.

Exact reviewed hashes:

| File | SHA256 |
| --- | --- |
| scripts/audit_sepclip_closed_loop_update_v2.py | 2b6dd11213f0aa28a1d6472124c8863e8ca96ba190229e6a1eace4ea1c815e99 |
| tests/test_sepclip_closed_loop_update_v2.py | 6a8c71a3296e48ec63902340b4ee1199170961fa02361b0d7840e31c84023b94 |
| tests/test_sepclip_closed_loop_nonlinear_jvp_v2.py | e3c98ee189b2bb33b47ee058cbb90d353d9bc7bc73bec62a58066e9e25d8b646 |
| Preserved scripts/audit_sepclip_closed_loop_update.py | 761e5ffd86d7432cdc47c21c9ad3a77a8c9d0f50ea39c40b608defdd812cef65 |
| Preserved tests/test_sepclip_closed_loop_update.py | 58f02e6bb956b0f08c1a385838e29fb8a4b7a6012352901cf039a1e62c166808 |
| Preserved tests/test_sepclip_closed_loop_nonlinear_jvp.py | 58fe105aa6b605cc3a194efa8c90bc91c6a00b20ae28aed62197e26f34b9db34 |

Inspection used source diffs, selected source reads, SHA256 and AST syntax/count parsing only. Static expanded counts are 19 V2 contracts and four real saved-chord contracts, total23. Parameter expansions were counted structurally, including NaN/Inf literals and two snapshot names crossed with two checkpoint steps. No pytest collection or numerical run occurred.

V2 lines116–144 replace only the directional adapter. Matching complete actor parameter keys, finite FP32 CUDA shapes and observation rows are required. Each old bank[0] parameter and each of the two tangents is detached and cloned with contiguous memory format. clone allocates private storage even when an input is already a contiguous view; detach alone or contiguous alone would not supply that ownership guarantee. All three dictionaries are explicit arguments to the compiled fullgraph core. The mathematical evaluation remains actor(old,x), D_actor(old,x)[first_delta], D_actor(old,x)[endpoint_delta], and the unchanged agent old-critic value. torch.func.functional_call is strict, so missing actor keys cannot silently fall back to live actor parameters. The actual allowlisted v12 actor has no extra buffers requiring substitution. No parameter projection, dtype change, finite-difference replacement, eager fallback, or disabled derivative is introduced.

The wrapper accepts strided observations, validates them before the graph and passes contiguous rows. It checks every returned primal/JVP/value for finiteness. Actor primal/tangent storage is private; the captured agent critic remains the unchanged old critic object, as before. The production diagnostic does not optimize/mutate that agent during the physical oracle. This is actor-bank ownership rather than a promise that arbitrary later caller mutation of the whole agent leaves its critic unchanged.

The new ownership/layout contract independently constructs both contiguous views and strided views for primal banks/tangents, also supplies a noncontiguous observation view, checks known first/end affine head shifts and then mutates caller arrays. Exact unchanged returned outputs falsify retention of those caller arrays. The source materializes the copies before compilation; the test therefore addresses the claimed storage ambiguity rather than merely inspecting is_contiguous. These tests still need their queued CUDA execution.

The added body-only test perturbs first hidden gate weights, requires nonzero directional output, and compares to two independently compiled centered primal evaluations. It also verifies endpoint linearity at twice the tangent. This tests actual hidden nonlinear dependence; it is no longer solely a bias-shift demonstration. The one h=.5, 5% relative gate is a useful adapter contract, not the complete physical numerical certificate.

The four additional real saved-chord tests provide the stronger numerical contract: unchanged original eight snapshot/update-trace byte pins, Adam/Anvil at4M/8M, exact old/first/end bank and chord ties, genuinely nonzero motion in both gate/up/down branches and readout/gain, independent compiled primals, all original eight finite-difference step sizes, rounded parameter-motion diagnostics, and two adjacent sizes meeting the unrelaxed original3% norm/.999cosine/2%chord/3%conservative-rounding-heuristic criteria. The V2 nonlinear test diff changes only its documentation/module dispatch/source-provenance path. A passing JVP vector contract would still not certify a small cancellation-dominated score-weighted PG sign.

The driver diff changes only its header and compiled_directional. Therefore reconstruction, actual old/first/end forward policy, seven-arm timing, complete state restoration, common inverse-CDF noise, action clipping/score law, frozen per-state observation/reward normalization, FP32 per-reward normalized law, raw/discounted accumulation, finite/GAE recursion, old fixed terminal value, summaries and CLI are source-identical to the frozen original. Their existing objective/sampling limitations remain; V2 is not a new causal estimator or a correction to those limitations.

The failure described for11795 is consistent with a Torch fake-forward-AD view/layout tracing problem. Private contiguous parameter/tangent copies and explicit graph arguments are a reasonable targeted attempt. Static inspection cannot determine the exact compiler root cause or guarantee that internal forward-AD views compile successfully. Gate on all23 executed contracts and exact executed source hashes, retain the original failed11795 and skipped11812/11825, and preserve queued11882/11883 pins. No original or unrelated file/job was changed by this reviewer.

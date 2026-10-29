# Closed-loop V2 JVP layout/capture repair preparation

Owned11795 failed its original queued CUDA contract run:12 passed,2 failed. Selective `mlq logs11795` reads locate both failures in compiled_directional at F.linear fake-tensor tracing, with forward differentiable-view tangent-layout assertion `!this_view_meta->has_fw_view()`. Stderr read was empty; the relevant pytest failure and summary were in stdout. No eager or model reproduction was run by this reviewer. Original11795, skipped11812/11825, queued11882/11883 and their source pins remain untouched.

| File | SHA256 |
| --- | --- |
| NEW scripts/audit_sepclip_closed_loop_update_v2.py | 2b6dd11213f0aa28a1d6472124c8863e8ca96ba190229e6a1eace4ea1c815e99 |
| NEW tests/test_sepclip_closed_loop_update_v2.py | 6a8c71a3296e48ec63902340b4ee1199170961fa02361b0d7840e31c84023b94 |
| NEW tests/test_sepclip_closed_loop_nonlinear_jvp_v2.py | e3c98ee189b2bb33b47ee058cbb90d353d9bc7bc73bec62a58066e9e25d8b646 |
| Preserved scripts/audit_sepclip_closed_loop_update.py | 761e5ffd86d7432cdc47c21c9ad3a77a8c9d0f50ea39c40b608defdd812cef65 |
| Preserved tests/test_sepclip_closed_loop_update.py | 58f02e6bb956b0f08c1a385838e29fb8a4b7a6012352901cf039a1e62c166808 |
| Preserved tests/test_sepclip_closed_loop_nonlinear_jvp.py | 58fe105aa6b605cc3a194efa8c90bc91c6a00b20ae28aed62197e26f34b9db34 |

V2 is a copied standalone version. Physical policy banks/arms, inverse-CDF law, state restoration, frozen row normalization, score law, GAE transport, finite guards and CLI behavior are preserved. Only the directional compiler/storage adapter and version labels changed. The old source's captured `bank[0]` primal is a view even when its strides report contiguous. Calling contiguous() alone may preserve that view. V2 materializes private detached clones with contiguous memory format for old actor parameters and both saved tangent dictionaries, then passes all three dictionaries as explicit arguments to the compiled fullgraph JVP core. The public callable still accepts observations and returns old logits,first JVP,endpoint JVP,old critic values. No eager fallback, skipped derivative path or altered gradient law is introduced.

Observed failure is a Torch fake-forward-AD view-layout assertion. Captured primal view topology versus separate tangent storage is a plausible triggering condition; source inspection alone does not establish the exact Torch root cause or prove the repair executes successfully. Explicit storage ownership and graph inputs remove those ambiguities while preserving the mathematical derivative. Queued CUDA reproduction remains necessary.

Static expanded contract count:19. All original14 contracts are retained against V2, including physical identical-policy coupling and matched GAE identities. New contracts exercise both contiguous-view and strided-view actor primals/tangents plus strided observations; verify known head-affine JVPs; show caller storage mutation cannot retarget already constructed private directional banks; compare a body-only nonlinear gate JVP to independently compiled centered primal chords; and reject nonfinite observations before graph execution. No tests or models were executed. AST syntax and static case counting are valid.

Additional copied NEW nonlinear_jvp_v2 test dispatches V2 and preserves every substantive numerical contract of the original skipped11825 test. All eight real v12 snapshot/update trace byte pins are AST-verified identical. Four cases cover Adam/Anvil4M/8M and both first/end full saved actor chords per case, require genuine motion in every hidden branch and readout, compare independent compiled primals, retain all eight h values and require two adjacent sizes satisfying the original error<.03,cos>.999,rounded-chord error<.02 and output-rounding heuristic<.03 gates. Those thresholds were not relaxed or chosen after new results. Combined static expanded count is23 (19 V2 contracts plus4 real-chord cases); real captured-chord validation is necessary before directional physical interpretation. Geometry review requested on the new nonlinear test.

Independent read-only physics review requested on the NEW hashes. Readiness means prepared for a separately pinned MLQ contract submission after review, not a completed causal result or successful compiled derivative. Frozen originals and all failed attempts remain preserved.

# First-gradient bridge V1 preparation receipt

Status: prepared and AST checked only; 23 expanded CUDA contracts are unexecuted. No models, tests, training, or queue submissions were executed by the author. Frozen sources and existing audits remain unchanged.

## Final preparation pins

| File | SHA256 |
| --- | --- |
| scripts/audit_sepclip_prescription_first_gradient_v1.py | 0adfa1acf004b26fcc4cfd5c429ac9056260e2be747eb347aed84699506b0261 |
| tests/test_sepclip_prescription_first_gradient_v1.py | 9e0d01920ecccd47ba2503443290573f971230e4b1e6ef97c760ee29e5a728d3 |
| scripts/sepclip_prescription_replay.py (unchanged) | 7edfc432775a0098a60d63cb3c8642c65a599c3e8ae8fa2cc2f3354885d1dcf9 |
| frozen v14 trainer | 27a3e7a84e2c5d0ab1679f84cc4ddb8dad5eb1b6e2bc5f66850ad93ccf1012b0 |
| frozen v15 trainer | 62bb7ca6a87072414ef9fa90f31a7e5e9a14762bec0633789bef5167ccbaec9a |
| source_manifest_v14.json (unchanged) | 80b4adf1e5301573084b28b3183d1f32dd96b8ed745785f0741c3932b249bbe2 |

Preliminary preparation pins before complete-gradient/device/stride checks and layout-corruption coverage: source ff8a8f3e3e34724fc1d70f7de639d914aae754fb8ed66fb7d0201e307c94af93; test af5a7b09eae761506a511df5305626f66a7511b0aaa6f8f94ac84d359def8a2b (22 cases). These are preparation history, not executed results.

## Acceptance protocol

Strict existing replay first binds the allowlisted actual v14/v15 module, Args, execution/manifest, snapshot and trace hashes, recorded old alpha/beta/log-probability/value, normalization, and endpoints. The new audit makes a private deep copy of the old agent and restores the independently saved pre-rollout shuffle RNG into a private CUDA generator. The pinned trainer uses this generator only in device_minibatches after rollout, so its first full-batch permutation is reconstructible. The saved state, permutation, post-first state, and caller RNG bytes are hashed.

The actual module's ppo_loss is compiled with the actual Args compile_mode, fullgraph=True and dynamic=False. Forward, zero_grad, backward, and exactly one foreach clip per identity-disjoint actor/critic trunk follow frozen source order. EVERY actor and critic clipped gradient must match the recorded first gradient bitwise, including complete names, shapes, dtype, device and stride, before any covector interpretation. Caller model, saved shuffle and caller CPU/current-device CUDA RNG are preserved, including exception paths through RNG isolation.

A separate compiled actor-only functional gradient evaluates the exact negative maximum/clamp PPO objective on the same permutation and advantage preprocessing. It retains source threshold tie derivatives, actual actor gain and action scale. Entropy coefficient must be zero and trunks must be disjoint. The report compares this ascent covector with negative recorded clipped LOSS gradients using full-vector norms, cosine, unconstrained least-squares common scale and residual, then parameter/weight+bias routes and residuals under the FULL actor common scale. Null covectors have undefined scale/cosine rather than invented ratios. No numerical residual threshold certifies independent equivalence.

## Contract scope

23 cases = four actual-source v14/v15 architecture/advantage fixtures + twelve capture/schema/entropy/shuffle/order/advantage corruptions + one independent source-permutation/RNG case + three permutation-semantic failures + three analytic full-vector/null/trunk-clipping witnesses. Actual-source fixtures require nonzero actor and critic gradients. The layout corruption preserves numerical values but must fail the recorded-gradient schema gate. Fixtures mostly use ratios near one; mixed-sign saturation and exact clip-boundary coverage is not independently demonstrated by these fixtures, although the implemented objective retains maximum/clamp tie semantics.

CLI: `--snapshot --trace --snapshot-sha256 --trace-sha256 --module-name --manifest-path --manifest-sha256 --execution-path --execution-sha256 --exp-name --output`. All inputs are required; module-name uses the frozen replay allowlist. Runtime records the new audit and frozen loader hashes and rechecks artifact/source provenance after measurement. Queue manifests must additionally pin all actual imported source dependencies.

Raw gradients/preclip norms were not saved and remain source-reconstructed diagnostics. Common LS scale is not an independently recorded clipping multiplier. Independent logits were not saved and are not invented by inversion. Stored PPO/GAE credit is not physical PG, global efficiency, optimizer superiority or benchmark causal attribution. Actual capture gradient equality and the future MLQ CUDA gate remain required before empirical use.

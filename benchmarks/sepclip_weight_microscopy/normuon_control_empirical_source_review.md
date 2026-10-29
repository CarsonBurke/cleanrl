# Completed NorMuon control: independent metadata review

Job 11811 succeeded once with exit code 0 and maxParallelRuns 1, after successful parity gate 11810. Fresh v12 control reproduces the historical v4 NorMuon-fast trajectory exactly on 28 shared scalar series, including all recorded episodic returns and lengths, policy, loss, gradient, learning-rate and matrix diagnostics. Timing/SPS differs, as expected for separate executions. This is same-seed deterministic execution reproduction, not an independent-seed replication or evidence of optimizer superiority.

| Metric | Fresh v12 | Historical v4 |
|---|---:|---:|
| Primary, inclusive steps [7,700,000, 7,900,000], 201 episodes | 8072.715564025575 | 8072.715564025575 |
| Final 100 episodes | 8189.44580078125 | 8189.44580078125 |
| Trailing 200k at 1M, 201 episodes | 1533.1255497718926 | 1533.1255497718926 |
| Trailing 200k at 2M, 201 episodes | 3598.2753749866392 | 3598.2753749866392 |
| Trailing 200k at 4M, 201 episodes | 5784.35836807175 | 5784.35836807175 |

There are 7,964 identical return events, ending at step 7,995,008. The final-100 window is recorded precisely in the accompanying JSON. These are raw episodic-return training scalars, not a separate deterministic evaluation. The primary uses an explicit trailing window through scripts/_runs.RunScalars; its symmetric window_mean helper was not used.

Execution settings and TensorBoard Args agree: HalfCheetah-v4, seed 1, 16 environments, 1,024 rollout steps, one full-batch minibatch, ten epochs, norm_adv=False, separate actor/critic clipping at 0.5, CUDA compiled reduce-overhead. The actual matrix method is NorMuon, learning rate .005, momentum .85, beta2 .95, bf16, actor only. The actor_optimizer=anvil2 label selects the matrix-plus-Adam dispatch; it does not select the matrix method. Six actor body matrices receive NorMuon; the actor head and remaining actor scalars, plus the entire critic, receive fused Adam (.0024, eps 1e-5). The fresh-only slow_clock=rollout metadata is inactive in the NorMuon branch. Shared logged Args otherwise differ only by experiment name; historical Args omit those two dispatch/clock fields.

The intended additive v14 job 11880 matches these NorMuon scalar settings exactly. Its actual selected body matrices also include the free input projection and two additive blocks (seven body matrices in total; the actor constructor explicitly uses n_blocks=2), so matching settings does not equate its architecture, constraints or parameter membership with this control.

All 79 source pins and three environment-source pins in the fresh execution receipt match current bytes. Relevant trainer SHA256 is b8063ba4c3806d7812d6422c1b4c531d11def5ed58e1a85ccd10cd4591d66867. Gate 11810 logged one passing CUDA contract; its test compares compiled gradients, parameters, warm matrix moments and auxiliary Adam moments over twenty synthetic updates against frozen v4, including differing irrelevant Anvil clocks. Its execution receipt and test pin are preserved in the JSON.

The JSON records exact run/event paths and hashes, full recorded Args, every shared scalar comparison, execution receipts, and six fresh checkpoint artifact paths, sizes and byte hashes. Checkpoint contents were deliberately not loaded: tensor, optimizer and normalizer content validation is outside this metadata review. Historical execution-time source pins were not located by selective filename search; current frozen v4 is pinned by the fresh execution/parity receipts. Exact historical scalar agreement and fresh pinned execution establish a reliable same-seed score control, without establishing physical PG reliability or a cause of the sub10k versus20k gap.

No models, GPU work, tests or jobs were executed/submitted during this review. Failed attempts and frozen files were preserved.

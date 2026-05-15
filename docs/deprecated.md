# Deprecated Components

## BiBoNoiseExpert (Removed May 14, 2026)

**What it was:** A routed expert that added Gaussian noise (std=0.5) to token representations. Ran at both training and inference.

**Why it was removed:**

1. **No academic backing.** No published paper validates a stochastic noise expert inside an MoE layer for decoder-only causal LMs. The concept is novel but unsubstantiated.

2. **Adjacent evidence is weak/negative:**
   - NEFTune (arXiv:2310.05914) shows noise helps at the *embedding layer* during *fine-tuning only* — not at FFN output during pretraining.
   - arXiv:2505.13500 shows noise injection into hidden states *degrades* model safety and accuracy.
   - arXiv:2602.08287 shows even small activation perturbations can reduce task accuracy by up to 40 points.

3. **MoE++ (ICLR 2025 Oral, Skywork AI) didn't need it.** The closest prior work on zero-computation experts (arXiv:2410.07348) uses Zero + Copy (Identity) + Constant experts — no stochastic expert. They achieved better performance + 1.1-2.1× throughput.

4. **Identity expert already covers the "dump bucket" use case.** If a token doesn't need processing, route to Identity. Deterministic, gradient-friendly, same effect without signal corruption.

5. **Inference non-determinism is a real cost.** Eval benchmarks, reproducibility, and debugging all suffer from a noise source in the forward pass.

6. **Theoretical equivalence to ridge regularization (arXiv:2102.07379) doesn't justify selective application.** The theory proves noise ≈ L2 regularization when applied *uniformly* and *infinitely often*. Applying it to router-selected tokens only, at inference too, is two unvalidated choices stacked.

**What replaced it:** The expert slot was converted to an additional MLP expert (n-3 MLPs instead of n-4). The model now has 3 special experts: Identity, Zero, ReLU².

**If you want noise regularization:** Apply it uniformly during training at the embedding layer (NEFTune-style) as a separate training technique, not as a routed expert.

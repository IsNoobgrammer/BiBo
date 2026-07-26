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

**What replaced it:** The expert slot was converted to an additional MLP expert. (The special-expert set has changed twice since: ReLU² became a regular PolyGLU activation, and the Zero expert was removed Jul 26 2026 in favor of a ±Identity pair — see the Zero-expert section below.)

---

## BiBoZeroExpert (Removed July 26, 2026)

**What it was:** A routed, param-free expert whose output was identically `0`. Intended as "learned suppression" — the router could choose to contribute nothing for a token.

**Why it was removed:**

1. **Its weight gradient is structurally zero.** `∂L/∂w_zero = ⟨grad_out, E_zero(x)⟩ = ⟨grad_out, 0⟩ = 0`. There is no signal that could ever tell the router "picking Zero was the right call," because Zero contributes nothing to compare against.

2. **The only gradient it does receive is negative.** Under `norm_topk_prob` softmax the top-k weights are coupled, so `∂L/∂s_zero = Σ_{i≠zero} ⟨grad_out, E_i⟩·(−w_i·w_zero) ≠ 0` — which is purely "picking Zero cost me the other experts' contribution." The score is monotonically pushed down.

3. **Consequence: all Zero usage was forced by the load balancer, against the gradient.** The aux-loss-free bias pushes tokens toward under-used experts; for Zero that push is always fighting the loss. Local ablations found Zero more harmful than Identity, consistent with this.

4. **The one production model with this architecture doesn't have it.** LongCat-Flash (arXiv:2509.01322, Eq. 1) defines its "zero-computation" experts as `E_i(x_t) = x_t` — **identity, not zero**. "Zero-computation" means zero FLOPs, not zero output. It runs Z=256 identity experts against N=512 FFN experts, with roughly 4 of 12 slots going to pass-through.

**What replaced it:** a signed pair — `+Identity` (`+w·x`) and `−Identity` (`−w·x`), both param-free, both gated by the router score, counted by `special_expert_pairs` per type. Because `norm_topk_prob` softmax forces all weights positive, a single Identity can only ever *add*; the pair restores a signed gate. It also **spans Zero's behavior as the `w₊ ≈ w₋` cancellation case** — same "skip this layer" effect, but reached by routing with live gradients on both branches.

**Known limitation:** softmax over sigmoid-squashed scores bounds the max/min weight ratio at `e ≈ 2.72`, so at `top_k=2` the weights sit near 0.5/0.5 and `(w₊ − w₋) ≈ 0` whenever a token picks both signs. The signed gate has real magnitude mainly when one sign is selected alongside GLU experts; expect it to work better at higher `top_k`.

**If you want noise regularization:** Apply it uniformly during training at the embedding layer (NEFTune-style) as a separate training technique, not as a routed expert.

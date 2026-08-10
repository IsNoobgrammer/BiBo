# BiBo roadmap

Candidate axes, with the question each one has to answer before it earns a GPU-hour.

## Read this first: the gate every item has to clear

Measured Aug 9 2026 on the bibo-baseline-2k stack (657M total / 110M active, 10 layers, 64x4x1024,
2000 steps): **between-seed noise is 0.0217 val**, and the kernel-rewrite effect at fixed seed is
0.0127 val / 0.1388 winogrande. See `bibo-noise-floor` in agent memory.

Consequences, learned the expensive way in the AttnRes carry round:

- **An n=1 arm cannot resolve anything under ~0.04 val.** The whole carry-scale round (sigmoid /
  raw / tanh / rms, spread 0.026) measured noise and was parked with no result.
- **Throughput and memory ARE resolvable** at 1.5%. Every real win this project has banked
  recently came from the kernel side, because those are direct measurements, not comparisons.
- So for each item below, ask: *is the expected effect bigger than 0.04 val, or is it measured in
  tok/s and GB?* If neither, it needs 3 seeds (~5 GPU-hours per comparison) or it should not run.
- **Stamp both repo commits at launch.** A residual_add rewrite mid-round silently invalidated the
  arm that anchored the entire carry conclusion.

Rough tiers below: **[speed]** measurable now, **[big]** plausibly >0.04, **[needs-seeds]** likely
inside the noise, **[infra]** unblocks everything else.

---

## 0. Infrastructure (do first -- these are why arms get wasted)

- [ ] **[infra] Measurement protocol.** Decide the standard: 3 seeds per arm for any loss claim,
      or explicit "unresolved" labelling. Without this every item below is a coin flip.
- [ ] **[infra] Expose architecture as CLI flags.** `hidden_size`, `num_hidden_layers`,
      `num_attention_heads`, `num_key_value_heads`, `intermediate_size` are hardcoded in
      `ablate/common/configs.py:SHARED`. Scaling currently needs a source edit per arm, which is
      exactly how the kernel-rewrite confound got in.
- [ ] **[infra] `mlp_only_layers = [0, 9]` is hardcoded to a 10-layer stack.** At 20 layers this
      makes layer 9 dense *mid-stack* and leaves the last layer MoE. Must become `[0, L-1]`.
- [ ] **[infra] `max_position_embeddings = 2048`** while the corpus is 4096-packed and the context
      ablation probes 4095. Already inconsistent at current scale.
- [ ] **[infra] `bias_update_threshold` is an absolute token count** (2621440 = 10 steps at
      64x4x1024). Changing batch or seq silently changes the update cadence.
- [ ] **[infra] Checkpoint durability.** `--hf_repo` exists and is unused; two boxes died this
      session and took their checkpoints with them.

## 1. Scaling

- [ ] **[big] muP / scale-invariant parameterization.** Tune LR once at small width and transfer.
      Without it every scale-up needs its own `--muon_lr` sweep, and the optimum has already moved
      once (3e-3 -> 1e-2). This is the highest-leverage item if the plan is to scale repeatedly.
- [ ] **[big] Scale DEPTH, not width.** 20-24 layers at matched active params. Depth is the axis
      with a specific open question (AttnRes has nothing to do at 10 layers; K3 runs 93), and
      effects on this project historically SHRINK with width/tokens (act-1b: 3.6x smaller at 1B).
- [ ] **[needs-seeds] Re-tune `muon_lr` / `adam_lr` at the new scale** if muP is skipped.

## 2. The user's list

1. [ ] **MTP (multi-token prediction).** Not scaffolded -- `num_nextn_predict_layers` exists only
       in `legacy/tests/`. Prior is AGAINST it on loss at 110M active (Meta's MTP work found it
       *hurts* small models, helps ~3B+; DeepSeek-V3 runs it at 671B; K3 ships it off). **But the
       speculative-decoding payoff is measured in decode tok/s, which is resolvable.** Decide the
       motivation first: quality (argue against) or inference speed (worth building). **[speed]**
       if judged on decode, **[needs-seeds]** if judged on loss.
2. [ ] **Better CE.** Concretely: **focal loss** `(1-p_y)^gamma * CE` -- keys on `p_y` directly,
       shift-invariant, one scalar per token, essentially free in the fused CE. This is the correct
       instrument for "let uncertain tokens learn more". NOT logit softcapping: that keys on
       absolute logit magnitude, which softmax is deliberately blind to, and it damps the
       confidently-WRONG gradient (the largest useful signal) as hard as the confident-correct one.
       Also consider **z-loss** (`log^2 Z`) if logit drift ever becomes a real symptom -- cheaper and
       better-posed than softcapping. No symptom observed so far: `|g|` sits at 0.09-0.20 late in
       training and QK-norm is already on. **[needs-seeds]**
3. [ ] **fp8 / fp4 training.** Expert GEMMs first (largest FLOP share, 34% of step time). Blackwell
       has native support. **[speed]** -- measured in tok/s and GB, so resolvable. Needs a parity
       gate: see `kernel-contract-bit-identity` (bit identity on the model dtype, not "close").
4. [ ] **Optimizer memory: Ember / AdamW state compression.** Adam m+v is ~5.3 GB at 657M.
       **[speed]** -- resolvable as peak-memory, which buys batch size, which buys tok/s.
5. [ ] **Gated attention.** Output gate on attention (K3 ships `mla_use_output_gate: true`).
       Cheap, and K3-proven at 93 layers. **[needs-seeds]**
6. [ ] **Differential attention.** Two softmax maps subtracted to cancel common-mode noise.
       Doubles attention params/compute. **[needs-seeds]**
7. [ ] **Decayed router noise.** Exploration early, deterministic late. Interacts with the existing
       `bias_update_factor` load balancing -- do not run both axes at once. **[needs-seeds]**
8. [ ] **Shared expert at scale.** `--n_shared` already exists (currently 0). K3 uses
       `num_shared_experts: 2`. Cheap to test, K3-proven. **[needs-seeds]**
9. [x] **Final AttnRes form** -- **PARKED Aug 9 2026.** Default is `b3s1 + carry + per_dim + raw`,
       chosen on TPS (1.5% over b3s2), NOT quality: every matched-code comparison is inside the
       seed floor (carry vs no-carry +0.0163, sigmoid vs raw +0.0081, raw vs tanh +0.0054).
       See `attnres-carry-round` in memory. Revive only by scaling depth.
10. [ ] **Fully NoPE on global layers.** K3 ships `mla_use_nope: true`. Pairs naturally with 11.
        **[needs-seeds]**
11. [ ] **Full/partial RoPE on SWA layers.** Currently `PARTIAL_ROPE = 0.334` everywhere. The
        NoPE-global + RoPE-local split is the standard modern pattern. Test 10 and 11 together as
        one arm, not two -- they are one design. **[needs-seeds]**
12. [ ] **Better `rope_theta`.** Currently 10000.0 with `max_position_embeddings` 2048 and a
        4096-packed corpus. Must be revisited if context is extended. **[needs-seeds]**
13. [ ] **QK-norm scaling of the q vector.** A learnable scale on the normalized q.
        `swa_qk_norm` is already on. **[needs-seeds]**
14. [ ] **Different tokenizer.** Vocab 81920 (QTK-81K). A tokenizer change moves the loss scale
        itself, so it is NOT comparable to any existing number -- needs a fresh baseline. Judge on
        bytes/token and downstream, never on val loss. **[big]** but incomparable.
15. [ ] **Online Shampoo.** Second-order; Muon is already an approximation of the same idea, so the
        marginal gain over Muon is the real question, not the gain over Adam. **[needs-seeds]**
16. [ ] **QK-Clip / MuonClip.** Clips QK logits to bound attention entropy collapse. Note this is
        the same problem QK-norm solves, and QK-norm is already on -- so the arm is *QK-Clip
        instead of QK-norm*, not in addition. **[needs-seeds]**

## 3. Additional suggestions

- [ ] **[big] MLA (latent KV compression).** K3: `kv_lora_rank 512`, `q_lora_rank 1536`. Cuts KV
      cache substantially. Measured in GB and decode tok/s -> **resolvable**, and K3-proven.
      Probably the highest-value item on this whole page after the infra block.
- [ ] **[big] Hybrid linear attention.** K3 runs KDA linear attention on ~74 of 93 layers with full
      attention every 4th. Big throughput/memory win at long context. Large build.
- [ ] **[speed] Train at seq 4096 directly.** The corpus is already 4096-packed and we truncate to
      1024. Changes what the extrapolation metrics even mean.
- [ ] **[needs-seeds] Data mixing ratio.** hi35/en65 is inherited, never ablated. Data effects are
      often larger than architecture effects at this scale.
- [ ] **[needs-seeds] Router: `noaux_tc` / grouped top-k.** K3's router method vs our bias-based
      load balancing.
- [ ] **[needs-seeds] Attention sinks / register tokens.** Cheap, well-established.
- [ ] **[speed] Untie word embeddings at scale.** `tie_word_embeddings: True` currently; at larger
      hidden the trade changes.

## Closed / do not re-run

- **AttnRes carry scale** (sigmoid/raw/tanh/rms) -- null, see item 9.
- **Slope family** (`slope05`/`slope15`) -- implemented and smoked (`f1ac1a0`), never run. The
  hypothesis it tested (slope-at-init as an effective LR on c) died with the ordering it explained.
- See agent memory for the longer list: activation axis, top-k depth, identity experts, MoE output
  norm, LatentMoE, router gate axis, NS coefficient axis.

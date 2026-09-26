# TODO

1. [ ] MTP
2. [ ] enchaning ce to be more effective ; maybe
3. [ ] fp8/4 training -- GATED on the optimizer round; full notes in "Quantized training" below
4. [ ] ember for adamw and more memory saving
5. [ ] gated attention
6. [ ] differential attention
7. [ ] decayed router noise
8. [ ] shared expert at scale ?
9. [ ] final version of attn-res (carry per dim + raw/rms/sigmoid/tanh) etc..
10. [ ] Fully NoPE on global layer
11. [ ] Fully/partial RoPE on swa
12. [ ] Better RoPE theta
13. [ ] QK_norm scaling q_vector -- marin_moe 67B runs a single scalar `qk_mult = 1.57` on the
        QK logits, tuned enough to be in the run name (`qk157`). Cheapest form of this axis:
        one scalar, no per-head parameters. Sweep {1.0, 1.25, 1.57, 2.0} before anything richer.
14. [ ] trying out different tokenizer
15. [ ] Online Shampoo optimizer maybe ?
16. [ ] QK-Clip Muon

## suggestions

17. [ ] MLA -- latent KV compression (K3: kv_lora_rank 512, q_lora_rank 1536)
18. [ ] muP -- tune LR at small width, transfer to large
19. [ ] hybrid linear attention (K3: KDA on ~74 of 93 layers, full attn every 4th)
20. [ ] train at seq 4096 directly (corpus is already 4096-packed, we truncate to 1024)
21. [ ] data mixing ratio -- hi35/en65 inherited, never ablated
22. [ ] router noaux_tc / grouped top-k
23. [ ] attention sinks / register tokens
24. [ ] untie word embeddings at scale
25. [ ] expose hidden_size / num_hidden_layers as CLI flags (hardcoded in configs.py SHARED)
26. [x] mlp_only_layers -- CLI flag, and SHARED default is now [] (no dense FFN anywhere). The old
        "wins by 0.066 val" claim did NOT survive its seed control -- see timeline/06.
27. [ ] max_position_embeddings = 2048 vs a 4096-packed corpus
28. [ ] --hf_repo unused; two boxes died with their checkpoints

## from the marin_moe 67B open run (Aug 24 2026)

Their config: 67B total / 2B active, 26 layers, hidden 2560, **256 experts top-4**, expert
intermediate 1280 (= hidden/2), **shared expert of 2560 on every layer**, SWA **2048** with
19 sliding / 7 full, RoPE theta 10k everywhere (NOT NoPE -- our split is ours), Muon with
wd 0.1 and norms routed to Adam, and **no aux loss applied** (`router_z_loss_coef = 0`,
`aux_loss_weighted = 0`) while still holding normalised routing entropy at 0.984-0.993.
Independent confirmation of three of our choices: all-MoE with no dense FFN, bias-only balancing,
and Muon + Adam-for-vectors + wd 0.1.

29. [x] log per-tensor params/norm and grad/norm -- DONE, ablate/common/tensor_health.py.
        grad/norm_min_over_tensors is the inert-parameter detector we have needed twice.
30. [x] log per-layer router z-loss (router LOGIT scale) -- DONE, same module. Diagnostic only,
        never added to the objective. Theirs climbs 3.2 -> 26.3 with depth while entropy stays flat.
31. [x] per-layer expert-load HISTOGRAM, not just max-load + entropy. Both summary stats hide a
        bimodal load; marin logs all 256 bins per layer.
32. [ ] per-domain bpb (marin logs Paloma across ~14 domains, plus macro_bpb vs bpb). We purged
        eval and now have a single val number, so we cannot tell a broad win from a one-domain win.
33. [ ] analytic flops/token for MFU. We pass --peak_tflops 480 by hand, so our MFU shifts when the
        architecture changes; theirs computes flops_per_token_analytic.
34. [ ] shared expert at scale -- see #8. They run one on EVERY layer at 2x expert width. We tested
        shared experts at 64 experts and dropped them; at 256 experts top-4 the calculus differs.
35. [ ] SWA window size. Ours is 128, theirs 2048 with fewer full layers (27% vs our 40%). Window
        size has never been swept here.

## from the L0 geometry round (Aug 31 - Sep 17 2026, bibo-dense-vs-moe-2k)

36. [x] per-layer expert geometry -- `--moe_override L:E:k:w`, both model paths (src/ and exp/),
        param counter and checkpoint loader aware of it.
37. [x] L0 DECISION: all-active 8x576 ensemble, MoE 64/top-6 elsewhere. ADOPTED in SHARED and
        pinned by a test. Tie on train, 68.4M smaller, predictable long-context -- but WORSE mean
        delta_ctx4095 (0.161 vs 0.085). Revisit if 4k serving becomes the target.
38. [ ] router z-loss coefficient. Top logits reach ~200 under a sigmoid gate, so the leading
        experts are saturated and gradient-free. `router_z_loss_coef` is 0. Cheapest open lead.
39. [ ] length-matched routing swap: feed 4095 of context but force L0 to its 1024-context expert
        choices. If CE recovers, routing that SHIFTS with length is the extrapolation cost.
        Checkpoint-only, no training. ablate/tools/length_probe.py is most of it.
40. [ ] `--val_seqs 2` makes val/loss noise -- it swung 0.0455 across seeds on one config. Raise it
        or drop it from the log line; rank on train loss + ctxabl (32 rows) meanwhile.
41. [ ] sm120 norm-router asserts E is a power of two. Blocks E=6/12/24 geometries; either lift it
        in the kernel or document it at the flag.
42. [ ] per-layer router GRAD norm. grad/norm/* is RMS-pooled across layers, so "L0's router is not
        learning" vs "not getting gradient" needed a local rebuild to tell apart.
43. [ ] shared expert (#8/#34) is now the best candidate to win back long-context on top of the
        adopted stack: a dense path every token sees, plus the sparse pool. `--n_shared 1`.

Seed floors measured this round (2000-step board, per-stack, never transfer):
  train 0.0016-0.0043 across seeds, 0.0004 same-seed nondeterminism
  ctx1024 ~0.007 across seeds | delta_ctx4095 0.0175 (ensemble) but 0.152 (sparse L0 -- unstable)

## Optimizer round (Sep 26 2026, bibo-aurora-vs-muown) -- finish BEFORE quantization

Goal: a baseline optimizer + schedule + wd, and which of them fits QAT best.
Board config, seed 23, 2000 steps, `--val_seqs 2` (kept for comparability with the first 3 runs).

Done:
- aurora (wd 0.1) vs muown (wd 0) vs switch aurora->muown@200 (muown wd 0), cosine.
  Final val: aurora 3.6010, muown +0.024, switch +0.026 (~1 seed floor). Muown/switch LEAD mid-run
  (switch -0.031..-0.042 at steps 600-900); the lead erodes smoothly with the cosine lr and is gone
  by ~1600. Both muown arms end at the same deficit -> it is the muown phase under annealing.
- Weight norms: aurora grows ~2x after warmup then shrinks 5-8% under wd; muown pins them near
  init (row norm = per-row gain g, which barely moves); switch pins at the step-200 value.
  Muown's 2.5-6x higher grad norm is pure 1/||W|| scale, NOT instability.
- `health/update_ratio/<group>` (||dW||/||W|| per step = effective step size) now logged.

44. [ ] ATTRIBUTION (queued Sep 26 14:46): aurora-wsd, muown-wsd (`--scheduler wsd --decay_frac 0.2`),
        muown-wd0.1 (cosine). Reading: if muown's lead holds through the WSD stable phase and
        flips only in the decay window -> it is the anneal (muown wants a different decay); if it
        erodes at constant lr -> norm / wd dynamics over time; if muown-wd0.1 keeps its lead
        through cosine -> weight decay. Cross-check with update_ratio (aurora self-decays via
        norm growth, muown's effective schedule is the nominal one).
45. [ ] confirm the winner on seed 42069 before adopting.
46. [ ] muown lr is unbracketed (improves as lr drops at 60 steps; mid-run lead at 1e-2).
47. [ ] then pick the QAT-facing optimizer: muown pins weight AND activation scale (residual rms
        0.1-0.4 vs aurora 10-42, max |x| 18 vs 752) -- a candidate advantage against the
        activation-norm drift seen in QAT (see below), unproven.

## Quantized training (QAT / low-precision pretraining) -- after the optimizer round

Why: loss-per-FLOP, not tps or memory. Target: same loss at ~1.5-1.7x less compute, or quantization
is not justified. Side benefit: 4-bit expert weights make 256 experts (vs 64) affordable.

What we know (Sep 26 2026, PTQ on the three 2k checkpoints; tools: ablate/tools/quant_probe.py,
ablate/tools/quant_probe_experts.py; the eager-expert probe reproduces the fused path exactly):
- 8-bit is ~free one-shot: int8/fp8 per-channel and MXFP8 W+A all <= +0.003.
- 4-bit one-shot (CE increase; A = attention + expert GEMM inputs, router/lm_head bf16):
    W4A16 nvfp4 +0.019 | W4A8 nvfp4/mxfp8 +0.021 | W4A4 nvfp4 +0.034 (+H16 +0.030) | W4A4 mxfp4 ~2x nvfp4.
  NVFP4 >> MXFP4. Most of the 4-bit cost is the WEIGHTS. Muown loses 0.001-0.007 less everywhere.
- Weights are Gaussian-clean in every run (row max/rms ~3.25, kurtosis ~0).
- gate_up input (post-norm hidden) is clean: amax/med ~7, nvfp4 underflow ~6.8% = Gaussian floor.
- down_proj input (act(gate)*up) is HEAVY-TAILED in all runs: amax/med 320-1425, nvfp4 underflow
  21-34%, mxfp4 29-46%; Hadamard-16 -> ~5% / ~8%. Radial normsilu does NOT tame it; muown no better.
- CAVEAT (important): these are ONE-SHOT numbers on bf16-trained models. They do not show error
  ACCUMULATION: in training the quantization noise biases every gradient and drifts (activation norms,
  block scales, underflow rates) compound over thousands of steps. Only a twin run shows it.

Speed arithmetic (verify on the box): only the GEMM share of a step speeds up. Blackwell FP4xFP8
runs at the FP8 rate, so for TRAINING W4A8 == W8A8 in speed and loses more -> W8A8 dominates on
loss/FLOP; W4 only buys memory (-> more experts). 1.5-1.7x needs FP4 on BOTH operands in fprop,
dgrad AND wgrad for most layers (e.g. 55% GEMM share: FP8 ~1.38x, FP4 ~1.70x).

Known recipes against accumulation (verify before building on them):
- NVIDIA NVFP4 pretraining (2025, 12B / ~10T tokens): 1D 16-blocks for activations and gradients,
  2D 16x16 blocks for weights (W and W^T quantize identically fwd/bwd), random Hadamard transform
  on wgrad inputs, stochastic rounding on gradients, last ~15% of layers kept bf16.
- DeepSeek-V3 FP8: 1x128 activation tiles, 128x128 weight blocks, fp32 accumulator promotion every
  128 K (FP8 tensor-core accumulation was too narrow on Hopper; check sm120).
- Twitter thread (Pollux, Sep 2026, qwen3-51m): tensorwise int8 QAT -> activation norms grow 2-45x,
  needs activation clamping; blockwise-1D activation quant converges, blockwise-2D does not (cross-token
  scales); W4A4 down_proj underflow climbs 0.2 -> 0.55 over training without Hadamard, ~0.08 with H16;
  SwiGLU clamping (gpt-oss / DeepSeek-V4 style, limit 3) fixed int8 w8a8 QAT; up-shift act(gate)*(up+1)
  gave a small real bf16 win (dead-product fix) but larger activations.

48. [ ] GEMM share of our step (ablate/tools/step_profile.py) -> the real speedup ceiling, decides
        W8A8 vs FP4.
49. [ ] sm120 capability + microbench: torch._scaled_mm / _scaled_grouped_mm, torchao MXFP8/NVFP4 MoE
        training, Triton tl.dot_scaled (does it take NVFP4 e4m3 block-16 scales on sm120?), accumulator
        precision. This is the compiled baseline our kernel must beat in speed AND drift.
50. [ ] QAT harness, emulated numerics first: experts fprop/dgrad/wgrad quantized per the recipes,
        plus a `quant/` W&B section logged every log step, per layer and GEMM input: underflow rate,
        overflow/saturation rate, block-scale underflow/saturation, input rms/amax, weight bin-flip
        rate. 2k TWIN vs bf16, same seed -- read the SLOPE of gap(t), not a snapshot.
51. [ ] arms: W8A8 MXFP8 first (proven), then NVFP4 W4A4 (+H16 / RHT on wgrad, SR on grads, 2D weight
        blocks); optimizer from the round above; watch activation-norm drift (muown vs aurora).
52. [ ] only then custom Triton kernels for the recipe that holds (fused quantize + block-scaled GEMM),
        beating the compiled baseline on speed AND on long-run underflow/overflow.
53. [ ] then the scale-up question: 256 experts at 4-bit weights vs 64 at bf16, matched memory.
54. [ ] candidate model-side fixes if drift appears: SwiGLU-style clamp on the down_proj input,
        up-shift (act(gate)*(up+1)), Hadamard on down_proj input only (the one tailed tensor).


# TODO

1. [ ] MTP
2. [ ] enchaning ce to be more effective ; maybe
3. [ ] fp8/4 training
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
26. [x] mlp_only_layers -- now a CLI flag (--mlp_only_layers), and the dense-vs-MoE round says
        `none` wins by 0.066 val at matched active params. Still hardcoded as the SHARED default.
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

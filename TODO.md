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
13. [ ] QK_norm scaling q_vector
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
26. [ ] mlp_only_layers = [0, 9] is hardcoded to 10 layers -- breaks on any depth change
27. [ ] max_position_embeddings = 2048 vs a 4096-packed corpus
28. [ ] --hf_repo unused; two boxes died with their checkpoints

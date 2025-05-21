### WiP BiBo 


### 1. Robust MoE Load Re-balancing

*   **Goal:** Achieve balanced utilization across Mixture-of-Experts (MoE) layers to prevent expert overloading and improve training stability and performance.
*   **Mechanism:** Implements a robust load re-balancing strategy using:
    *   **Noisy Top-k Gating:** Introduces small noise (`router_noise`) during expert selection in training to encourage exploration and prevent certain experts from being consistently favored.
    *   **Tokens-Processed Heuristic:** Incorporates a global loss rebalancing component based on expert usage. This involves auxiliary loss terms calculated based on the fraction of tokens processed by each expert and the router's confidence scores. The router bias term is updated periodically (`bias_update_factor`, `bias_update_threshold`) based on token processing history to dynamically adjust expert preference.
    *   **Normalized Top-k Probabilities:** Optionally normalizes probabilities before selecting the top-k experts (`norm_topk_prob=True`).
*   **References:**
    *   [Deepseek's paper on heuristic based load balancing](https://arxiv.org/abs/2408.15664)
    *   [Qwen's blog on global load balancing](https://qwenlm.github.io/blog/global-load-balance/)

### 2. Dynamic Compute and Enhanced Pathways in FFN

*   **Goal:** Allow the model to dynamically allocate computational resources per token based on complexity *and* significantly increase the number of potential processing pathways within the FFN layers, enhancing representational power without proportional parameter increase.
*   **Mechanism:** The expert pool now includes:
    - **(n-4) MLP Experts:** Standard, parameterized MLPs.
    - **Identity/Residual Expert:** Passes input unchanged (enables residual/skip routing).
    - **Zero Expert:** Outputs zeros, implemented as `x * 0` for robust device/dtype/sharding propagation.
    - **Noise Expert:** Adds Gaussian noise (std=0.5) to input for stochastic routing/regularization.  
      ⚠️ **Highly experimental**: Not recommended for production or official validation. Intended only for private/ablation testing.
    - **ReLU² Expert:** Applies `relu(x)^2` nonlinearity.
    - **Order:** All are selectable by the router; shared expert is always active.
*   **Dynamic Compute:** The router can select any of these experts per token, including skipping computation (identity/zero) or adding noise for regularization.
*   **Increased Combinatorics:** These diverse options enable a combinatorial explosion of token pathways, improving specialization and efficiency.
*   **Device/Distributed Safety:** All experts are implemented to be device/dtype/sharding-agnostic for maximal compatibility.
*   **References:**
    *   [OLMoE: Why not Shared Expert?](https://arxiv.org/html/2409.02060v1#S4.F6)
    *   [MoE++ Residual experts](https://arxiv.org/abs/2410.07348)
    *   [DeepSeek-V2/V3 MoE scaling rationale](https://kexue.fm/archives/10945#%E6%AF%94%E4%BE%8B%E5%9B%A0%E5%AD%90)

### 3. MoE Configuration and Shared Expert Scaling

*   **Shared Expert Scaling (`moe_shared_scaling`):**
    - To balance the routed and shared expert outputs, a scaling factor λ is applied to the shared expert output.
    - If left at the default (`1.0`), λ is automatically estimated at runtime using a statistical simulation (see [DeepSeek-V2/V3, Muon](https://kexue.fm/archives/10945#%E6%AF%94%E4%BE%8B%E5%9B%A0%E5%AD%90)) based on your MoE config (`num_routed_experts`, `num_experts_per_tok`, `num_shared_experts`).
    - The chosen λ is printed at model init for transparency and can be overridden manually.
    - This ensures routed/shared outputs have comparable norms, improving training stability and expert utilization.
*   **Expert Pool Example:**
    - With `num_routed_experts=16`, you get 12 MLPs, 1 identity, 1 zero, 1 noise, 1 relu² expert (all routed), plus 1 shared convolution expert.
*   **Robustness:**
    - All experts propagate device, dtype, and sharding correctly.
    - Zero expert uses `x * 0` pattern for full compatibility.
    - Noise expert uses `torch.no_grad` .

### 4. Convolution-Based Router

*   **Goal:** Improve router decisions by incorporating local context, rather than relying solely on the current token's features.
*   **Mechanism:** Offers an optional causal 1D convolution-based router (`router_type="conv"`). This router uses a kernel (`kernel_size`) to consider a local window of preceding token representations when calculating routing scores. This allows the router to make more informed decisions based on the immediate history.
*   **Reference:** Inspired by concepts in [https://arxiv.org/abs/2209.10655](https://arxiv.org/abs/2209.10655) (Note: BiBo uses a simpler local convolution compared to the EMA approach in the paper).

### 4. Convolution-Based Shared Expert

*   **Goal:** Enhance the representational capacity of the shared FFN expert by incorporating local context.
*   **Mechanism:** The shared expert FFN (`num_shared_experts=1`) uses a causal 1D convolution with a defined kernel size (`kernel_size`). This allows the transformation within the shared expert to consider a local window of token features, potentially capturing sequential patterns more effectively.  Its side-effects are mitigated by Point 2. (i.e more combinatrics for a token ; even if it has shared expert)
*   **Reference:** Conceptual alignment with ideas discussed in [https://x.com/ZeyuanAllenZhu/status/1918684280001089619](https://x.com/ZeyuanAllenZhu/status/1918684280001089619)

### 5. Activation-Based Normalization
*   **SCRAPED (Doesn't provide what it claims)**
*   **References:**
    *   [Tanh proposed](https://arxiv.org/abs/2503.10622)
    *   [Proof of why?](https://arxiv.org/abs/2503.21708) (Theoretical basis)

### 6. Scalable Softmax (SSMax)


*   **Goal:** Improve attention mechanism stability and performance, especially for very long sequences, by preventing attention scores from becoming overly diffuse ("fading") or overly concentrated.
*   **Mechanism:** Implements SSMax (`use_ssmax=True`), a modification to the standard softmax function used in attention.
    *   **Core Idea:** It scales the attention weights *before* the softmax operation based on the sequence length (`k_len`).
    *   **Scaling Factor:** The scaling factor `C` is calculated as `s * log(k_len)`, where `s` is a learnable parameter (`ssmax_scale`) per head, and `log(k_len)` adapts the scaling to the sequence length (clamped at a minimum length of 2 to avoid issues with log(1) or log(<1)).
    *   **Effect:** This scaling effectively acts as a learnable, sequence-length-adaptive temperature applied *before* the softmax. Comparing the ratio of two softmax probabilities:
        *   Standard Softmax Ratio: `exp(z_i) / exp(z_k) = exp(z_i - z_k)`
        *   SSMax Ratio: `exp(C * z_i) / exp(C * z_k) = exp(C * (z_i - z_k)) = (exp(z_i - z_k))^C`
        The scaling factor `C` exponentiates the standard ratio, allowing the model to learn how sharply the attention should focus based on the context length.
    *   **Benefit:** Helps maintain meaningful attention distributions over longer sequences where standard softmax might struggle.
*   **Reference:** [https://arxiv.org/abs/2501.19399](https://arxiv.org/abs/2501.19399)



*   [ ] **MLA (Multi-Latent Attention):** Explore attention mechanisms like those in DeepSeek-V2. (Ref: [DeepSeek-V2 Paper](https://arxiv.org/abs/2404.19753))
*   [ ] **Advanced Positional Embeddings:** Investigate alternatives like VO-RoPE or NoPE. (Ref: [kexue.fm Blog Post](https://kexue.fm/archives/10862))
*   [ ] **MLKV (Multi-Layer Key/Value Sharing):** Implement sharing of KV projections/caches across layers. (Ref: [Twitter/X Link](https://t.co/AmoqdyLiod))
*   [ ] **Meta-Tokens:** Experiment with adding latent tokens for unsupervised meta-learning.
*   [ ] **KAN as Routable Expert:** Consider integrating Kolmogorov-Arnold Networks (KAN) as an option within the MoE FFN layers.
*   [ ] **Wavelet-like RoPE**
*   [ ] **QK-Normalization:** Evaluate the trade-offs of QK-Norm, especially its interaction with SSMax for long-context performance. (Ref: [arxiv:2501.18795](https://arxiv.org/pdf/2501.18795))
*   [ ] **Other Innovations:** Seek low-compute, high-performance improvements compatible with the current framework.





### Pretraining-Tips

- Add CoTs with self-correction like ds-r1 in pretrain corpus ; helps model self-correct without RL (ref: https://physics.allen-zhu.com/part-2-grade-school-math/part-2-2)
- Use more hidden_layers than hidden_dims ; for same param (depth is better than width) (ref: https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)
* add intruct tuned data to pretrain (should be diverse and generalizd) (else not so much perf. improvements) (ref: https://physics.allen-zhu.com/part-3-knowledge/part-3-1) 
* * i.e add qa in pretrain data along with its biography ; results shows it learns from qa then bio and then generalizes on oout of distribution(ood) bio and then ood qa
* * it implies that in pretrain ; we should teach the model how to use its knowledege ; can't just train on corpus of internet slop ; need structed data ;  i think even comprehension and answering would help. if factual include popular data for more pronounced effect
* At min can train till int8(tpu go brr)/fp8 ;  (ref: https://physics.allen-zhu.com/part-3-knowledge/part-3-3)
* Pretrain data quality matters **alot** ; if anyhow can't filter then use some keytags to differentiate text ; i.e use wikipideia.org before starting a text sample  (ref: https://arxiv.org/abs/2404.05405)
* for data to params ; i thinks its more in the ratio of 32 bits per params and considering tokenizer's avg. compression as 2bits/token then for 7b it would be 112B pure/clean/knowledgeable deta (depends only on full param size on activated ; for moe its kind 0.95% capacity of dense of same total param (but i think that moe should have more capacity due to token's combinatrics of path to follow)  )


# Awknokewledhement 

कर्मण्येवाधिकारस्ते मा फलेषु कदाचन्।
मा कर्मफलहेतुर्भूर्मा ते संगोस्त्वकर्मणि।।

हम कोई काम करने से पहले उसके फल के बारे में सोचते हैं और फिर काम करते हुए भी नतीजे के बारे में सोचते रहते हैं। जब काम की वजह से मनचाहा फल मिलता है, तब हम खुश होते हैं। इससे अहंकार बढ़ जाता है लेकिन अगर हमारी इच्छा के मुताबिक फल न मिले तो हम दुखी हो जाते हैं। 

इंसान को केवल कर्म का ही अधिकार है, उसके फल के बारे में चिंता करने का नहीं। इसलिए तुम कर्मों के फल की चिंता मत कर और कर्म से विमुख मत हो


**Our Nation**

| Language  | Name for India        |
| :-------- | :-------------------- |
| Assamese  | ভাৰত (Bhārôt)         |
| Bengali   | ভারত (Bhārôt)         |
| Boro/Bodo | भारत (Bharot)         |
| Dogri     | भारत (Bhārat)         |
| English   | India                 |
| Gujarati  | ભારત (Bhārat)         |
| Hindi     | भारत (Bhārat)         |
| Kannada   | ಭಾರತ (Bhārata)        |
| Kashmiri  | भारत, हिन्दोस्तान (Bhārata, Hindōstān) |
| Konkani   | भारत (Bhārat)         |
| Maithili  | भारत (Bhārat)         |
| Malayalam | ഭാരതം (Bhāratam)      |
| Marathi   | भारत (Bhārat)         |
| Nepali    | भारत (Bhārat)         |
| Oriya     | ଭାରତ (Bhārata)        |
| Punjabi   | ਭਾਰਤ (Bhārat)         |
| Sanskrit  | भारतम् (Bhāratam)     |
| Santali   | ᱥᱤᱧᱚᱛ/ᱵᱷᱚᱨᱳᱛᱵᱳᱨᱥᱚ (Siñôt/Bhôrotborsô) |
| Sindhi    | ڀارت (Bharatu)        |
| Telugu    | భారతదేశం (Bhāratadēśam) |
| Urdu      | بھارَت (Bhārat)        |

## License

- Will kinda/mostly be Apache 2.0
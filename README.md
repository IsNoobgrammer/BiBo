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
*   **Mechanism:** This is achieved by including identity/residual experts alongside the standard routed experts (`num_routed_experts`).
    *   **Dynamic Compute:** The router can choose an identity expert for a token. This acts as a passthrough (residual connection), effectively skipping the heavy computation of a standard expert FFN for that token, saving computational resources on tokens deemed "simpler".
    *   **Increased Combinatorics:** Crucially, these identity experts are treated as selectable options by the router alongside the routed experts. This dramatically increases the total number of expert combinations a token might be processed by.
        *   *Example:* Consider an MoE with 32 experts where the router selects the top 6 (`k=6`). The number of combinations is C(32, 6).
        *   If we designate 1 expert as shared (always active) and route among the remaining 31, selecting 5 (`k=5`), the routed combinations become C(31, 5). The token is processed by these 5 + the 1 shared expert.
        *   Now, if we add 2 *parameter-free* identity experts to the pool of *selectable* experts (making the pool size 31 routed + 2 identity = 33), and still select 5 (`k=5`), the number of possible combinations for the routed part jumps to C(33, 5). The token is still processed by the selected 5 (which could include identity experts) + the 1 shared expert.
    *   This combinatorial explosion allows the model to learn much richer functions and specialize pathways more effectively, mitigating potential representational bottlenecks from using fewer, or shared, experts.
*   **Reference:** [OLMoE talks why it not took Shared Expert](https://arxiv.org/html/2409.02060v1#S4.F6) (Illustrates the concept of dynamic computation/conditional routing).

### 3. Convolution-Based Router

*   **Goal:** Improve router decisions by incorporating local context, rather than relying solely on the current token's features.
*   **Mechanism:** Offers an optional causal 1D convolution-based router (`router_type="conv"`). This router uses a kernel (`kernel_size`) to consider a local window of preceding token representations when calculating routing scores. This allows the router to make more informed decisions based on the immediate history.
*   **Reference:** Inspired by concepts in [https://arxiv.org/abs/2209.10655](https://arxiv.org/abs/2209.10655) (Note: BiBo uses a simpler local convolution compared to the EMA approach in the paper).

### 4. Convolution-Based Shared Expert

*   **Goal:** Enhance the representational capacity of the shared FFN expert by incorporating local context.
*   **Mechanism:** The shared expert FFN (`num_shared_experts=1`) uses a causal 1D convolution with a defined kernel size (`kernel_size`). This allows the transformation within the shared expert to consider a local window of token features, potentially capturing sequential patterns more effectively.  Its side-effects are mitigated by Point 2. (i.e more combinatrics for a token ; even if it has shared expert)
*   **Reference:** Conceptual alignment with ideas discussed in [https://x.com/ZeyuanAllenZhu/status/1918684280001089619](https://x.com/ZeyuanAllenZhu/status/1918684280001089619)

### 5. Activation-Based Normalization

*   **Goal:** Approximate standard normalization layers (like RMSNorm) with a faster alternative.
*   **Mechanism:** Replaces traditional normalization with scaled activation functions (`layer_norm_type="dyt"` or `"erf"`). Based on research suggesting normalization layers behave similarly to scaled activations, this approach uses functions like Tanh and Erf to achieve normalization effects. This method is computationally cheaper (~50% faster than RMSNorm) as it avoids calculating variance and can be applied pointwise.
*   **References:**
    *   [Tanh proposed](https://arxiv.org/abs/2503.10622)
    *   [Proof of why?](https://arxiv.org/abs/2503.21708) (Theoretical basis)

### 6. Scalable Softmax (SSMax)

*   **Goal:** Improve attention mechanism stability and performance, especially for very long sequences.
*   **Mechanism:** Implements SSMax (`use_ssmax=True`), a modification to the standard softmax function used in attention. It involves scaling the attention weights before the softmax operation, aiming to prevent attention scores from becoming too concentrated or too diffuse over long contexts. In a gist: a learnable, seq-len adaptive temperature applied per head to control attention sharpness, preventing fading in long contexts.

*   **Reference:** [lol a single guy ; cooking fire](https://arxiv.org/abs/2501.19399)




TODO: 
- MLA
- Better RoPE/NoPE ?? VO-RoPE etc.. ref: https://kexue.fm/archives/10862 
- MLKV (Multi layer kv/proj and cache sharing) ref: x/zedzed
- Meta-tokens (giving model a scratc-pad and allowing it to learn implicitly in unsupervised manner)
- KAN also as a routable expert in FFN ?? 
- Any other innovation : thing is low compute req. or very high perf. imrpov and compatibilty with current framework



# Awknokewledhement 

कर्मण्येवाधिकारस्ते मा फलेषु कदाचन्।
मा कर्मफलहेतुर्भूर्मा ते संगोस्त्वकर्मणि।।

हम कोई काम करने से पहले उसके फल के बारे में सोचते हैं और फिर काम करते हुए भी नतीजे के बारे में सोचते रहते हैं। जब काम की वजह से मनचाहा फल मिलता है, तब हम खुश होते हैं। इससे अहंकार बढ़ जाता है लेकिन अगर हमारी इच्छा के मुताबिक फल न मिले तो हम दुखी हो जाते हैं। 

इंसान को केवल कर्म का ही अधिकार है, उसके फल के बारे में चिंता करने का नहीं। इसलिए तुम कर्मों के फल की चिंता मत कर और कर्म से विमुख मत हो

Lets make India a frontier in AI ; via key innovation in arch thus reducing compute requirements and model's inherent capability

## License

- Will kinda/mostly be Apache 2.0
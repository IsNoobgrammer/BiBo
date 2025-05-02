# BiBo Model: Decoder Layer & MoE/MLP Design

## Key Differences from Standard Transformer Decoder

### 1. **Dense MLP and MoE Layer Structure**
- **BiBoMLP:**
  - Standard SwiGLU MLP used for dense layers (first and last layers by default, as set in config `mlp_only_layers`).
- **MLPExpert:**
  - SwiGLU-based MLP expert used in MoE layers (for routed experts).
- **ModifiedConvolutionalExpert:**
  - Causal 1D convolutional expert (usually as a shared expert in MoE layers).
  - Applies causal padding and convolution over sequence.
- **IdentityExpert:**
  - Pass-through expert (used as a fallback or for ablation).

### 2. **MoE Routing**
- **BiBoMoERouter:**
  - Computes routing weights for each token using a learned linear projection, temperature scaling, noise injection, and softmax.
  - Supports top-k expert selection per token (`num_experts_per_tok` from config).
  - Includes learnable bias and optional bias update based on token distribution.

### 3. **MoE Forward Pass**
- **BiBoMoELayer:**
  - Maintains a list of routed experts (MLPExpert + optional IdentityExpert) and shared experts (typically one ModifiedConvolutionalExpert).
  - For each token, routes to top-k experts and combines their outputs (weighted sum).
  - Shared expert output is added to routed expert output.
  - Supports bias update for router during training.

### 4. **Config Integration**
- All expert counts, hidden sizes, kernel sizes, router temperature/noise, and layer types are controlled by `BiBoConfig`.
- `mlp_only_layers` (from config) determines which layers are dense (BiBoMLP) and which are MoE (BiBoMoELayer).
- `num_routed_experts`, `num_shared_experts`, `moe_intermediate_size`, `kernel_size`, `router_temperature`, and other MoE parameters are set via config.

### 5. **Summary of Changes from Standard Decoder Layer**
- Flexible switching between dense and MoE layers per config.
- MoE layers support both routed (token-wise) and shared (convolutional) experts.
- Routing is dynamic and can include noise and bias updates for better load balancing.
- All architectural choices are exposed via config for easy experimentation.

---

## Example Usage
- To use a dense MLP for the first and last layers, set `mlp_only_layers=[0, num_hidden_layers-1]` in `BiBoConfig` (default).
- All other layers will use MoE blocks as described above.

---

## References
- SwiGLU: https://arxiv.org/abs/2002.05202
- MoE Routing: https://arxiv.org/abs/1701.06538, https://arxiv.org/abs/2101.03961
- Convolutional Expert: Custom, inspired by recent MoE/Conv hybrid research.

---

For more details, see the implementation in `BiBoMLP`, `MLPExpert`, `ModifiedConvolutionalExpert`, `BiBoMoERouter`, and `BiBoMoELayer` classes.

# BiBo Architecture Analysis & CodeRabbit Configuration

## Project Overview

**BiBo** is a research project implementing a PyTorch-based Mixture of Experts (MoE) transformer with the goal of comparing novel architectural hypotheses against established baselines (Qwen3 and Qwen3MoE).

### Target Hardware
- **Remote**: TPU v5e-8 (Kaggle Notebook, 16GB VRAM per chip, 2D mesh)
- **Local**: RTX 3050 Laptop (4GB VRAM) + Intel UHD Graphics (2GB VRAM)

### Key Requirement
All code must be **torch.compile()** and **torch-xla** compatible for TPU execution.

---

## Architecture Components

### 1. **BiBoConfig** (`src/configuration_bibo.py`)
Configuration class with **feature toggles** for all experimental features.

**Key Design Principle**: Every new feature MUST have a config toggle with validation.

**Notable Features**:
- `attention_type`: Toggle between `softmax`, `sliding_window`, `linear`, `gdn`, `kda`
- `use_ssmax`: Scaled Softmax (learnable per-head scaling for long sequences)
- `router_type`: Toggle between `mlp` and `conv` routing
- `moe_shared_scaling`: Auto-computed scaling factor for shared expert output
- `num_routed_experts`: Includes MLP experts + special experts (Identity, Zero, Noise, ReLU²)
- `num_shared_experts`: Typically 1 Conv expert
- `bias_update_factor` & `bias_update_threshold`: Heuristic load balancing parameters

**Validation**: All parameters validated in `__init__` to catch errors early.

---

### 2. **BiBoAttention** (`src/modeling_bibo.py`)
Multi-head attention with:
- **Grouped Query Attention (GQA)**: Multiple Q heads, fewer K/V heads
- **Multi-layer KV sharing (MLKV)**: Shared K/V projections across layer groups
- **Multiple attention variants**:
  - `softmax`: Standard attention with optional SSMax scaling
  - `sliding_window`: Local causal window attention
  - `linear`: Linear attention with cumulative KV state
  - `gdn`: Gated Delta Network (recurrent with learned gates)
  - `kda`: Key-Delta Attention (recurrent with per-dimension gates)

**Key Methods**:
- `eager_standard_attention()`: Standard softmax attention
- `eager_sliding_window_attention()`: Causal sliding window
- `eager_recurrent_attention()`: Unified implementation for linear/gdn/kda
- `_apply_ssmax_query_scaling()`: SSMax scaling (learnable, seq-len adaptive)

**Compilation Considerations**:
- Recurrent attention uses Python loop (may need torch.jit.script for compilation)
- All tensor operations are device-agnostic

---

### 3. **BiBoMoELayer** (`src/modeling_bibo.py`)
Mixture of Experts layer with:

**Routed Experts** (configurable count):
- MLP experts (standard feed-forward)
- Identity expert (residual connection)
- Zero expert (returns zeros, preserves sharding)
- Noise expert (adds Gaussian noise)
- ReLU² expert (ReLU activation squared)

**Shared Expert**:
- `BiBoCausalConv1D`: 1D causal convolution expert (always active)

**Key Features**:
- Heuristic load balancing via learnable router bias
- Threshold-based bias updates (accumulated TPE)
- Shared expert scaling for norm balancing

---

### 4. **BiBoMoERouter** (`src/modeling_bibo.py`)
Router with two modes:

**MLP Router**:
- Linear projection: `hidden_size → num_routed_experts`

**Conv Router**:
- Causal 1D convolution with local context
- Kernel size configurable

**Key Features**:
- Learnable bias for load balancing
- Router lambda scaling (Skywork-MoE style)
- Noise injection during training
- Top-k expert selection with normalized weights

---

## Baseline Models

### **baseline/qwen3** (Dense Transformer)
- Standard Qwen3 architecture
- Reference for dense model comparison
- Read-only (no modifications)

### **baseline/qwen3moe** (Sparse MoE Transformer)
- Qwen3MoE architecture with standard MoE
- Reference for sparse model comparison
- Read-only (no modifications)

**Purpose**: Establish performance baselines for BiBo hypothesis validation.

---

## CodeRabbit Configuration Updates

### Changes Made

1. **Profile**: Changed from `chill` to `assertive` for stricter ML research code review

2. **Request Changes Workflow**: Enabled to block merges on critical issues

3. **Path Filters**: Added filters for:
   - TensorBoard logs (`events.out.tfevents.*`)
   - Git internals
   - Legacy code (archived experiments)

4. **Knowledge Base**: Comprehensive project context including:
   - Architecture components
   - Baseline models
   - Feature toggles
   - Compilation requirements
   - Testing standards

5. **Review Instructions**: 10 critical review areas:
   - **Feature Toggles (MANDATORY)**: Every new feature needs config toggle
   - **Compilation Compatibility**: torch.compile() and torch-xla checks
   - **Tensor Operations**: Shape verification, device placement
   - **Numerical Stability**: Division by zero, overflow checks
   - **Memory Efficiency**: Unnecessary copies, gradient checkpointing
   - **Gradient Flow**: Detach calls, no_grad usage
   - **Baseline Compatibility**: No changes to baseline models
   - **Testing Requirements**: Tests for new features
   - **Documentation**: Config params, shape comments, design decisions
   - **Performance**: O(n²) operations, redundant computations

---

## Key Design Patterns

### 1. **Feature Toggle Pattern**
```python
# Config parameter
use_feature: bool = False

# Validation
if self.use_feature and not self.compatible_param:
    raise ValueError("use_feature requires compatible_param")

# Conditional logic
if self.use_feature:
    output = self.feature_forward(x)
else:
    output = self.standard_forward(x)
```

### 2. **Compilation-Safe Pattern**
```python
# BAD: Python control flow on non-tensor
if hidden_size > 1024:
    output = large_model(x)

# GOOD: Tensor-based control flow
mask = (hidden_size > 1024).float()
output = mask * large_model(x) + (1 - mask) * small_model(x)
```

### 3. **Device-Agnostic Pattern**
```python
# BAD: Hardcoded device
x = x.cuda()

# GOOD: Use input device
device = x.device
y = torch.zeros_like(x)  # Inherits device
```

### 4. **Numerical Stability Pattern**
```python
# BAD: Division by zero risk
output = numerator / denominator

# GOOD: Clamped denominator
output = numerator / denominator.clamp_min(eps)
```

---

## Testing Strategy

### Test Files
- `tests/conftest.py`: Transformers stub for dependency-free testing
- `tests/test_attention_variants.py`: Comprehensive attention mechanism tests

### Test Coverage
All attention variants must pass:
1. **Finite outputs and gradients**: No NaN/Inf
2. **Causality**: Future tokens don't affect past outputs
3. **Shape contracts**: Correct tensor dimensions
4. **Reference implementations**: Match manual calculations

---

## Next Steps for Contributors

### When Adding a New Feature:

1. **Add config parameter** in `BiBoConfig.__init__`:
   ```python
   use_new_feature: bool = False
   ```

2. **Add validation** in `BiBoConfig.__init__`:
   ```python
   if self.use_new_feature and self.incompatible_param:
       raise ValueError("use_new_feature incompatible with incompatible_param")
   ```

3. **Implement feature** with conditional logic:
   ```python
   if self.config.use_new_feature:
       output = self.new_feature_forward(x)
   else:
       output = self.standard_forward(x)
   ```

4. **Add tests** in `tests/test_*.py`:
   ```python
   def test_new_feature_finite_outputs():
       config = make_config(use_new_feature=True)
       # ... test implementation
   ```

5. **Document** in config docstring and inline comments

6. **Verify compilation**:
   ```python
   model = torch.compile(BiBoModel(config))
   # Should not raise errors
   ```

---

## CodeRabbit Review Checklist

When CodeRabbit reviews your PR, it will check:

- ✅ New features have config toggles
- ✅ Config parameters are validated
- ✅ No Python control flow in forward passes
- ✅ No dynamic shapes (or properly handled)
- ✅ Numerical stability (epsilon, clamp_min)
- ✅ Gradient flow preserved
- ✅ Device-agnostic code
- ✅ Tests for new features
- ✅ Documentation for config params
- ✅ No changes to baseline models

---

## Summary

The BiBo project is a well-structured research codebase with:
- **Clear separation** between hypothesis (src/) and baselines (baseline/)
- **Feature toggles** for all experimental features
- **Compilation-first** design for TPU execution
- **Comprehensive testing** for all attention variants
- **Automated review** via CodeRabbit with ML-specific rules

The updated CodeRabbit configuration ensures all PRs maintain these standards and catch common ML/TPU pitfalls early.

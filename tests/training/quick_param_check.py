"""Quick parameter count check for BiBo vs Qwen3MoE"""
import torch
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


print("="*80)
print("Parameter Count Verification")
print("="*80)

# BiBo config (8 experts: 1 identity, 1 noise, 1 zero, 1 relu, 2 MLP, 2 Conv)
print("\nCreating BiBo model...")
bibo_config = BiBoConfig(
    vocab_size=5000,
    hidden_size=512,
    intermediate_size=1536,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=2,
    num_routed_experts=8,  # 8 experts total
    num_experts_per_tok=2,
    moe_intermediate_size=512,
    router_type='conv',
    router_noise=0.01,
    router_lambda=1.0,
    kernel_size=3,
    mlp_only_layers=[0, 11],
    use_ssmax=True,
    max_position_embeddings=512,
)

bibo_model = BiBoForCausalLM(bibo_config)
bibo_params = count_parameters(bibo_model)

print(f"BiBo config:")
print(f"  Experts: {bibo_config.num_routed_experts}")
print(f"  Hidden: {bibo_config.hidden_size}")
print(f"  Layers: {bibo_config.num_hidden_layers}")
print(f"  Total params: {bibo_params[0]:,}")

# Qwen3MoE config - reduce experts to match BiBo param count
print("\nCreating Qwen3MoE model...")
qwen_config = Qwen3MoeConfig(
    vocab_size=5000,
    hidden_size=512,
    intermediate_size=1536,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=2,
    num_experts=6,  # Reduced to match BiBo params
    num_experts_per_tok=2,
    moe_intermediate_size=512,
    mlp_only_layers=[0, 11],
    max_position_embeddings=512,
)

qwen_model = Qwen3MoeForCausalLM(qwen_config)
qwen_params = count_parameters(qwen_model)

print(f"Qwen3MoE config:")
print(f"  Experts: {qwen_config.num_experts}")
print(f"  Hidden: {qwen_config.hidden_size}")
print(f"  Layers: {qwen_config.num_hidden_layers}")
print(f"  Total params: {qwen_params[0]:,}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
diff = abs(bibo_params[0] - qwen_params[0])
diff_pct = diff / qwen_params[0] * 100

print(f"BiBo:      {bibo_params[0]:>15,} params")
print(f"Qwen3MoE:  {qwen_params[0]:>15,} params")
print(f"Difference: {diff:>14,} params ({diff_pct:.2f}%)")

if diff_pct < 10:
    print(f"\n✓ Parameter counts are close enough ({diff_pct:.2f}% difference)")
    print("  Proceeding with benchmark is fair.")
else:
    print(f"\n⚠ Parameter counts differ by {diff_pct:.2f}%")
    print("  Consider adjusting configs for fairer comparison.")

print("="*80)

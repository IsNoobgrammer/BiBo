"""
BiBo vs Qwen3MoE attention parity test.
Verify identical attn impl (RoPE, GQA, norm, softmax).
"""
import torch
import pytest
from src.modeling.attn.base import BiBoAttention
from src.configuration_bibo import BiBoConfig
from baseline.qwen3moe.modeling import Qwen3MoeAttention
from baseline.qwen3moe.config import Qwen3MoeConfig


def make_causal_mask(batch_size, seq_len, device, dtype):
    """Causal mask: (batch, 1, q_len, kv_len)"""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)


@pytest.mark.parametrize("seq_len", [16, 64, 128])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_bibo_qwen_attention_parity(seq_len, num_heads, num_kv_heads):
    """
    Test: BiBo attn (SSMax off) == Qwen attn
    Same weights → same output
    """
    if num_heads % num_kv_heads != 0:
        pytest.skip(f"num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}")
    
    batch_size = 2
    hidden_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # BiBo config (SSMax disabled)
    bibo_cfg = BiBoConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=True,
        use_ssmax=False,
    )
    
    # Qwen config
    qwen_cfg = Qwen3MoeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=True,
    )
    
    # Models
    bibo_attn = BiBoAttention(bibo_cfg, layer_idx=0).to(device).to(dtype)
    qwen_attn = Qwen3MoeAttention(qwen_cfg, layer_idx=0).to(device).to(dtype)
    
    # Copy weights: Qwen → BiBo
    with torch.no_grad():
        bibo_attn.q_proj.weight.copy_(qwen_attn.q_proj.weight)
        bibo_attn.q_proj.bias.copy_(qwen_attn.q_proj.bias)
        bibo_attn.k_proj.weight.copy_(qwen_attn.k_proj.weight)
        bibo_attn.k_proj.bias.copy_(qwen_attn.k_proj.bias)
        bibo_attn.v_proj.weight.copy_(qwen_attn.v_proj.weight)
        bibo_attn.v_proj.bias.copy_(qwen_attn.v_proj.bias)
        bibo_attn.o_proj.weight.copy_(qwen_attn.o_proj.weight)
        if bibo_attn.o_proj.bias is not None and qwen_attn.o_proj.bias is not None:
            bibo_attn.o_proj.bias.copy_(qwen_attn.o_proj.bias)
        bibo_attn.q_norm.weight.copy_(qwen_attn.q_norm.weight)
        bibo_attn.k_norm.weight.copy_(qwen_attn.k_norm.weight)
    
    bibo_attn.eval()
    qwen_attn.eval()
    
    # Input
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Causal mask
    attn_mask = make_causal_mask(batch_size, seq_len, device, dtype)
    
    # Compute RoPE (use BiBo's - now Qwen-compatible)
    cos, sin = bibo_attn.rotary_emb(hidden_states, position_ids)
    position_embeddings = (cos, sin)
    
    # BiBo forward
    bibo_out, _, _ = bibo_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attn_mask,
    )
    
    # Qwen forward
    qwen_out, _ = qwen_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attn_mask,
    )
    
    # Check parity
    max_diff = (bibo_out - qwen_out).abs().max().item()
    mean_diff = (bibo_out - qwen_out).abs().mean().item()
    
    print(f"\n[seq={seq_len}, heads={num_heads}, kv_heads={num_kv_heads}]")
    print(f"  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    
    # Strict parity
    assert max_diff < 1e-5, f"Attn outputs differ: max_diff={max_diff:.2e}"
    assert mean_diff < 1e-6, f"Attn outputs differ: mean_diff={mean_diff:.2e}"
    print(f"  ✓ BiBo == Qwen (parity confirmed)")


if __name__ == '__main__':
    print("="*80)
    print("BIBO vs QWEN ATTENTION PARITY TEST")
    print("="*80)
    
    test_bibo_qwen_attention_parity(seq_len=64, num_heads=8, num_kv_heads=2)
    test_bibo_qwen_attention_parity(seq_len=128, num_heads=8, num_kv_heads=8)
    test_bibo_qwen_attention_parity(seq_len=16, num_heads=4, num_kv_heads=2)
    
    print("\n" + "="*80)
    print("✓ All parity tests passed - BiBo attn == Qwen attn")
    print("="*80)

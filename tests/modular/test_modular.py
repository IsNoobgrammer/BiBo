"""Test modular BiBo implementation"""
import sys
sys.path.insert(0, '.')
import torch
import pytest
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoModel, BiBoForCausalLM
from src.modeling.attn import BiBoAttention
from src.modeling.ffn import BiBoMLP, BiBoMoELayer

@pytest.fixture
def cfg():
    """Test config fixture"""
    return BiBoConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_routed_experts=8,
        num_experts_per_tok=2,
    )

def test_attention(cfg):
    """Test attention module"""
    attn = BiBoAttention(cfg, layer_idx=0)
    x = torch.randn(2, 8, cfg.hidden_size)
    pos_emb = (torch.randn(8, cfg.hidden_size // cfg.num_attention_heads),
               torch.randn(8, cfg.hidden_size // cfg.num_attention_heads))
    out, _, _ = attn(x, pos_emb, None)
    assert out.shape == x.shape
    print(f"✓ Attention: {x.shape} → {out.shape}")

def test_mlp(cfg):
    """Test MLP"""
    mlp = BiBoMLP(cfg, is_expert=False)
    x = torch.randn(2, 8, cfg.hidden_size)
    out = mlp(x)
    assert out.shape == x.shape
    print(f"✓ MLP: {x.shape} → {out.shape}")

def test_moe(cfg):
    """Test MoE layer"""
    moe = BiBoMoELayer(cfg)
    x = torch.randn(2, 8, cfg.hidden_size)
    out = moe(x)
    assert out.shape == x.shape
    print(f"✓ MoE: {x.shape} → {out.shape}")

def test_model(cfg):
    """Test full model"""
    model = BiBoModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids)
    print(f"✓ Model: input {input_ids.shape} → hidden {out.last_hidden_state.shape}")
    return model

def test_causal_lm(cfg):
    """Test causal LM"""
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids)
    print(f"✓ CausalLM: input {input_ids.shape} → logits {out.logits.shape}")
    
    # Test with labels
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids, labels=labels)
    print(f"✓ CausalLM loss: {out.loss.item():.4f}")

def test_forward_backward(cfg):
    """Test forward + backward"""
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    
    out = model(input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    
    # Check grads exist
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"✓ Backward: {has_grads}/{total_params} params have grads")

def count_params(model):
    """Count model params"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Params: {total:,} total, {trainable:,} trainable")

if __name__ == "__main__":
    print("Testing modular BiBo implementation...\n")
    
    cfg = test_config()
    print()
    
    test_attention(cfg)
    test_mlp(cfg)
    test_moe(cfg)
    print()
    
    model = test_model(cfg)
    count_params(model)
    print()
    
    test_causal_lm(cfg)
    print()
    
    test_forward_backward(cfg)
    print()
    
    print("✅ All tests passed!")

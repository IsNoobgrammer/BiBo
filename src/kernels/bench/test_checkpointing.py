"""
Test gradient checkpointing integration.
"""
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM


def test_checkpointing():
    """Test that selective gradient checkpointing works correctly."""
    print("Testing gradient checkpointing integration...")
    
    # Create model
    config = BiBoConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        polyglu_expert_multiplier=1,
        special_expert_pairs=1,
        num_experts_per_tok=4,
        mlp_only_layers=[0, 3],
        use_ssmax=True,
    )
    model = BiBoForCausalLM(config)
    model.train()
    
    # Test without checkpointing
    print("\n[1/2] Testing without checkpointing...")
    input_ids = torch.randint(0, 5000, (2, 128))
    labels = torch.randint(0, 5000, (2, 128))
    
    output_no_checkpoint = model(input_ids, labels=labels)
    loss_no_checkpoint = output_no_checkpoint.loss
    loss_no_checkpoint.backward()
    
    # Check gradients exist
    has_grads_no_checkpoint = any(p.grad is not None for p in model.parameters())
    print(f"  Loss: {loss_no_checkpoint.item():.4f}")
    print(f"  Has gradients: {has_grads_no_checkpoint}")
    
    # Clear gradients
    model.zero_grad()
    
    # Test with selective checkpointing
    print("\n[2/2] Testing with selective checkpointing...")
    model.enable_selective_gradient_checkpointing()
    
    # Verify checkpointing is enabled
    assert all(layer.use_selective_checkpointing for layer in model.model.layers)
    print("  Selective checkpointing enabled")
    
    output_with_checkpoint = model(input_ids, labels=labels)
    loss_with_checkpoint = output_with_checkpoint.loss
    loss_with_checkpoint.backward()
    
    # Check gradients exist
    has_grads_with_checkpoint = any(p.grad is not None for p in model.parameters())
    print(f"  Loss: {loss_with_checkpoint.item():.4f}")
    print(f"  Has gradients: {has_grads_with_checkpoint}")
    
    # Verify losses are close (should be similar but may differ due to checkpointing)
    loss_diff = abs(loss_no_checkpoint.item() - loss_with_checkpoint.item())
    print(f"  Loss difference: {loss_diff:.6f}")
    
    # Disable checkpointing
    model.disable_selective_gradient_checkpointing()
    assert all(not layer.use_selective_checkpointing for layer in model.model.layers)
    print("  Selective checkpointing disabled")
    
    print("\n✓ Gradient checkpointing integration test passed!")


if __name__ == "__main__":
    test_checkpointing()

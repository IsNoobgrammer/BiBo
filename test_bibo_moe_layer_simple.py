import torch
from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoMoELayer

def test_bibo_moe_layer():
    """
    A simple test to verify the basic functionality of BiBoMoELayer.
    """
    # Create a minimal config
    config = BiBoConfig(
        hidden_size=60,
        moe_intermediate_size=64,
        num_routed_experts=3,
        num_shared_experts=1,
        num_experts_per_tok=2,
        bias_update_factor=1e-3,
        bias_update_threshold=100,
        kernel_size=3
    )
    
    # Create the MoE layer
    moe_layer = BiBoMoELayer(config)
    
    # Test in both training and eval modes
    for is_training in [True, False]:
        if is_training:
            moe_layer.train()
        else:
            moe_layer.eval()
        
        # Create sample inputs of different batch sizes and sequence lengths
        for batch_size in [1, 2]:
            for seq_len in [5, 10]:
                # Create input tensor
                x = torch.randn(batch_size, seq_len, config.hidden_size)
                
                # Forward pass
                output = moe_layer(x)
                
                # Check output shape
                assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
                
                # Check output values are finite
                assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    print("All BiBoMoELayer tests passed!")

if __name__ == "__main__":
    test_bibo_moe_layer()
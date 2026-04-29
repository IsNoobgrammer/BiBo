import unittest
import torch
import torch.nn as nn
from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoMoELayer, BiBoMLP, BiBoIdentityExpert, BiBoCausalConv1D

class TestBiBoMoELayer(unittest.TestCase):
    def setUp(self):
        # Create a small config for testing
        self.config = BiBoConfig(
            hidden_size=60,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=1,
            num_experts_per_tok=2,
            bias_update_factor=1e-3,
            bias_update_threshold=100,
            router_temperature=1.0,
            router_noise=0.0,
            kernel_size=3
        )
        self.moe_layer = BiBoMoELayer(self.config)
        
        # Create sample input
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_dim = self.config.hidden_size
        self.input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

    def test_initialization(self):
        """Test that the MoE layer initializes correctly with the right number of experts."""
        # Check number of routed experts
        self.assertEqual(len(self.moe_layer.routed_experts), self.config.num_routed_experts)
        
        # Check that the last expert is an identity expert
        self.assertIsInstance(self.moe_layer.routed_experts[-1], BiBoIdentityExpert)
        
        # Check that other routed experts are MLPs
        for i in range(self.config.num_routed_experts - 1):
            self.assertIsInstance(self.moe_layer.routed_experts[i], BiBoMLP)
        
        # Check shared experts
        self.assertEqual(len(self.moe_layer.shared_experts_list), self.config.num_shared_experts)
        if self.config.num_shared_experts > 0:
            self.assertIsInstance(self.moe_layer.shared_experts_list[0], BiBoCausalConv1D)

    def test_forward_shape(self):
        """Test that the forward pass returns the correct shape."""
        output = self.moe_layer(self.input)
        self.assertEqual(output.shape, self.input.shape)

    def test_router_output(self):
        """Test that the router produces valid routing decisions."""
        # Get routing decisions directly from the router
        top_k_indices, top_k_weights = self.moe_layer.gate(self.input)
        
        # Check shapes
        self.assertEqual(top_k_indices.shape, (self.batch_size, self.seq_len, self.config.num_experts_per_tok))
        self.assertEqual(top_k_weights.shape, (self.batch_size, self.seq_len, self.config.num_experts_per_tok))
        
        # Check that indices are valid (within range of num_routed_experts)
        self.assertTrue((top_k_indices >= 0).all())
        self.assertTrue((top_k_indices < self.config.num_routed_experts).all())
        
        # Check that weights sum to approximately 1
        weight_sums = top_k_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))

    def test_bias_update(self):
        """Test that the bias update mechanism works correctly."""
        # Save initial bias
        initial_bias = self.moe_layer.gate.bias.clone()
        
        # Create an unbalanced token distribution
        tokens_per_expert = torch.tensor([100, 10, 10, 10], dtype=torch.float, 
                                         device=initial_bias.device)
        
        # Apply bias update
        self.moe_layer.update_bias(tokens_per_expert)
        
        # Check that bias for over-utilized expert decreased
        self.assertTrue(self.moe_layer.gate.bias[0] < initial_bias[0])
        
        # Check that bias for under-utilized experts increased
        self.assertTrue((self.moe_layer.gate.bias[1:] > initial_bias[1:]).all())

    # def test_expert_selection(self):
    #     """Test that experts are correctly selected and weighted."""
    #     # Mock the router to return predictable routing decisions
    #     def mock_router(hidden_states):
    #         batch_size, seq_len = hidden_states.shape[:2]
    #         # Route even tokens to experts 0 and 1, odd tokens to experts 2 and 3
    #         indices = torch.zeros(batch_size, seq_len, self.config.num_experts_per_tok, dtype=torch.long)
    #         weights = torch.ones(batch_size, seq_len, self.config.num_experts_per_tok) / self.config.num_experts_per_tok
            
    #         for i in range(seq_len):
    #             if i % 2 == 0:
    #                 indices[:, i, :] = torch.tensor([0, 1])
    #             else:
    #                 indices[:, i, :] = torch.tensor([2, 3])
            
    #         return indices, weights
        
    #     # Replace router function
    #     original_router = self.moe_layer.gate
    #     self.moe_layer.gate = mock_router
        
    #     # Run forward pass
    #     output = self.moe_layer(self.input)
        
    #     # Restore original router
    #     self.moe_layer.gate = original_router
        
    #     # Check output shape
    #     self.assertEqual(output.shape, self.input.shape)

    def test_shared_expert_contribution(self):
        """Test that the shared expert contributes to the output."""
        # Create a version without shared experts for comparison
        config_no_shared = BiBoConfig(
            hidden_size=60,
            moe_intermediate_size=128,
            num_routed_experts=4,
            num_shared_experts=0,
            num_experts_per_tok=2,
            bias_update_factor=1e-1,
            bias_update_threshold=100
        )
        moe_layer_no_shared = BiBoMoELayer(config_no_shared)
        
        # Copy the routed experts and router weights to ensure fair comparison
        for i in range(len(moe_layer_no_shared.routed_experts)):
            with torch.no_grad():
                for param_no_shared, param_with_shared in zip(
                    moe_layer_no_shared.routed_experts[i].parameters(),
                    self.moe_layer.routed_experts[i].parameters()
                ):
                    param_no_shared.copy_(param_with_shared)
        
        with torch.no_grad():
            moe_layer_no_shared.gate.bias.copy_(self.moe_layer.gate.bias)
            if hasattr(self.moe_layer.gate, 'gate_proj') and hasattr(moe_layer_no_shared.gate, 'gate_proj'):
                moe_layer_no_shared.gate.gate_proj.weight.copy_(self.moe_layer.gate.gate_proj.weight)
        
        # Run both models
        with torch.no_grad():
            output_with_shared = self.moe_layer(self.input)
            output_no_shared = moe_layer_no_shared(self.input)
        
        # Outputs should be different if shared expert contributes
        if self.config.num_shared_experts > 0:
            self.assertFalse(torch.allclose(output_with_shared, output_no_shared))

    def test_training_vs_eval_mode(self):
        """Test that the model behaves differently in training vs eval mode."""
        # Set to training mode with noise
        self.moe_layer.train()
        self.moe_layer.gate.router_noise = 0.5
        
        # Run forward pass in training mode
        torch.manual_seed(42)
        output_train = self.moe_layer(self.input)
        
        # Set to eval mode
        self.moe_layer.eval()
        
        # Run forward pass in eval mode
        torch.manual_seed(42)
        output_eval = self.moe_layer(self.input)
        
        # Outputs should be different due to noise in training
        self.assertFalse(torch.allclose(output_train, output_eval))

    def test_threshold_based_bias_update(self):
        """Test that bias updates occur after threshold is reached."""
        # Reset counters
        self.moe_layer.tokens_processed.zero_()
        self.moe_layer.accumulated_tpe.zero_()
        
        # Set to training mode
        self.moe_layer.train()
        
        # Save initial bias
        initial_bias = self.moe_layer.gate.bias.clone()
        
        # Process fewer tokens than threshold
        small_batch = torch.randn(1, self.config.bias_update_threshold // 2, self.hidden_dim)
        self.moe_layer(small_batch)
        
        # Bias should not have changed
        print("1 forward pass and still no threshold cross ; so bais no change ",self.moe_layer.gate.bias, initial_bias)

        self.assertTrue(torch.allclose(self.moe_layer.gate.bias, initial_bias))
        
        # Process more tokens to exceed threshold
        self.moe_layer(small_batch)
        print("2 forward pass and threshold cross ; so bais change ",self.moe_layer.gate.bias, initial_bias)
        self.moe_layer(small_batch)
        print("3 forward pass and no threshold cross ; no bais change ",self.moe_layer.gate.bias, initial_bias)
        self.moe_layer(small_batch)
        print("4 forward pass and threshold cross ; so bais change ",self.moe_layer.gate.bias, initial_bias)

        
        # self.moe_layer(small_batch)
        # self.moe_layer(small_batch)
        # self.moe_layer(small_batch)
        
        # Bias should have changed
        self.assertFalse(torch.allclose(self.moe_layer.gate.bias, initial_bias))

if __name__ == '__main__':
    unittest.main()
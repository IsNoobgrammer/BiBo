import unittest
import tempfile
import os
import json

from configuration_bibo import BiBoConfig

class BiBoConfigTest(unittest.TestCase):

    def test_default_initialization(self):
        """Tests initialization with default values."""
        config = BiBoConfig()
        self.assertEqual(config.vocab_size, 128000)
        self.assertEqual(config.hidden_size, 1536)
        self.assertEqual(config.intermediate_size, 4104)
        self.assertEqual(config.num_hidden_layers, 8)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.num_key_value_heads, 2)
        self.assertEqual(config.num_layer_kv_sharing, 2)
        self.assertEqual(config.num_meta_tokens, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.initializer_range, 0.02)
        self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(config.layer_norm_type, "rms")
        self.assertTrue(config.use_cache)
        self.assertTrue(config.use_ssmax)
        self.assertIsNone(config.pad_token_id)
        self.assertEqual(config.bos_token_id, 0)
        self.assertEqual(config.eos_token_id, 0)
        self.assertTrue(config.tie_word_embeddings)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertIsNone(config.rope_scaling)
        self.assertEqual(config.attention_dropout, 0.0)
        self.assertTrue(config.use_sliding_window)
        self.assertEqual(config.sliding_window, 512)
        self.assertEqual(config.max_window_layers, 8) # Defaults to num_hidden_layers
        self.assertFalse(config.attention_bias)
        self.assertEqual(config.decoder_sparse_step, 1)
        self.assertEqual(config.moe_intermediate_size, 512)
        self.assertEqual(config.num_routed_experts, 8)
        self.assertEqual(config.num_shared_experts, 1)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.num_experts, 9) # Defaults to routed + shared
        self.assertEqual(config.router_temperature, 1.3)
        self.assertEqual(config.bias_update_factor, 1e-2)
        self.assertEqual(config.bias_update_threshold, 100_000)
        self.assertEqual(config.router_noise, 0.5)
        self.assertEqual(config.router_type, "mlp")
        self.assertEqual(config.kernel_size, 3)
        self.assertFalse(config.norm_topk_prob)
        self.assertFalse(config.output_router_logits)
        self.assertEqual(config.mlp_only_layers, [0, 7]) # Defaults based on num_hidden_layers

    def test_custom_initialization(self):
        """Tests initialization with custom values."""
        custom_params = {
            "vocab_size": 50000,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 12,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_layer_kv_sharing": 3,
            "num_meta_tokens": 4,
            "hidden_act": "gelu",
            "max_position_embeddings": 2048,
            "initializer_range": 0.01,
            "rms_norm_eps": 1e-6,
            "layer_norm_type": "dyt",
            "use_cache": False,
            "use_ssmax": False,
            "pad_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "tie_word_embeddings": False,
            "rope_theta": 20000.0,
            "rope_scaling": {"type": "linear", "factor": 2.0},
            "attention_dropout": 0.1,
            "use_sliding_window": False,
            "sliding_window": 1024, # Should be ignored if use_sliding_window is False, but set for test
            "max_window_layers": 6,
            "attention_bias": True,
            "decoder_sparse_step": 2,
            "moe_intermediate_size": 1024,
            "num_routed_experts": 16,
            "num_shared_experts": 2,
            "num_experts_per_tok": 4,
            "num_experts": 18, # Explicitly set
            "router_temperature": 1.0,
            "bias_update_factor": 0.0,
            "bias_update_threshold": 50000,
            "router_noise": 0.0,
            "router_type": "conv",
            "kernel_size": 5,
            "norm_topk_prob": True,
            "output_router_logits": True,
            "mlp_only_layers": [1, 10],
        }
        config = BiBoConfig(**custom_params)

        for key, value in custom_params.items():
            self.assertEqual(getattr(config, key), value, f"Parameter {key} mismatch")

    def test_validation_hidden_size_divisible(self):
        """Tests validation for hidden_size divisibility."""
        with self.assertRaisesRegex(ValueError, "hidden_size .* must be divisible by num_attention_heads"):
            BiBoConfig(hidden_size=100, num_attention_heads=12)

    def test_validation_attn_heads_divisible_by_kv_heads(self):
        """Tests validation for num_attention_heads divisibility by num_key_value_heads."""
        with self.assertRaisesRegex(ValueError, "num_attention_heads .* must be divisible by num_key_value_heads"):
            BiBoConfig(num_attention_heads=12, num_key_value_heads=5)

    def test_validation_positive_values(self):
        """Tests validation for positive value constraints."""
        negative_params = [
            ("max_position_embeddings", -1),
            ("vocab_size", 0),
            ("rms_norm_eps", 0.0),
            ("initializer_range", 0.0),
            ("sliding_window", 0),
            ("moe_intermediate_size", -10),
            ("kernel_size", 0),
            ("router_temperature", 0.0),
        ]
        for param, value in negative_params:
            with self.subTest(param=param, value=value):
                with self.assertRaises(ValueError):
                    BiBoConfig(**{param: value})

    def test_validation_range_values(self):
        """Tests validation for range constraints."""
        range_params = [
            ("attention_dropout", -0.1),
            ("attention_dropout", 1.1),
            ("bias_update_factor", -0.1),
            ("router_noise", -0.1),
        ]
        for param, value in range_params:
            with self.subTest(param=param, value=value):
                with self.assertRaises(ValueError):
                    BiBoConfig(**{param: value})

    def test_validation_layer_norm_type(self):
        """Tests validation for layer_norm_type enum."""
        with self.assertRaisesRegex(ValueError, "rms_norm_type must be one of 'rms', 'dyt', or 'erf'"):
            BiBoConfig(layer_norm_type="invalid_norm")

    def test_validation_num_experts_per_tok(self):
        """Tests validation for num_experts_per_tok."""
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok cannot exceed total number of experts"):
            BiBoConfig(num_routed_experts=8, num_shared_experts=1, num_experts_per_tok=10) # 10 > 8+1

    def test_validation_mlp_only_layers_range(self):
        """Tests validation for mlp_only_layers indices."""
        with self.assertRaisesRegex(ValueError, "mlp_only_layers index .* is out of range"):
            BiBoConfig(num_hidden_layers=8, mlp_only_layers=[-1, 7])
        with self.assertRaisesRegex(ValueError, "mlp_only_layers index .* is out of range"):
            BiBoConfig(num_hidden_layers=8, mlp_only_layers=[0, 8])

    def test_serialization(self):
        """Tests saving and loading the configuration."""
        config = BiBoConfig(vocab_size=1000, hidden_size=120, num_hidden_layers=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(config_path))

            loaded_config = BiBoConfig.from_pretrained(tmpdir)
            self.assertEqual(config.to_dict(), loaded_config.to_dict())

    def test_max_window_layers_default(self):
        """Tests that max_window_layers defaults to num_hidden_layers."""
        config = BiBoConfig(num_hidden_layers=10)
        self.assertEqual(config.max_window_layers, 10)

    def test_num_experts_default(self):
        """Tests that num_experts defaults correctly."""
        config = BiBoConfig(num_routed_experts=16, num_shared_experts=4)
        self.assertEqual(config.num_experts, 20) # 16 + 4

    def test_mlp_only_layers_default(self):
        """Tests that mlp_only_layers defaults correctly."""
        config = BiBoConfig(num_hidden_layers=10)
        self.assertEqual(config.mlp_only_layers, [0, 9])


if __name__ == "__main__":
    unittest.main()

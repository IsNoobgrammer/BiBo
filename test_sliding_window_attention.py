import math
import torch
import pytest
from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoAttention


def test_window_size_1():
    """Test with window_size=1 (each token only attends to itself)."""
    batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8
    hidden_size = num_heads * head_dim

    # Create a proper config
    config = BiBoConfig()
    config.hidden_size = hidden_size
    config.num_attention_heads = num_heads
    config.num_key_value_heads = num_heads  # Same as num_heads to avoid divisibility issue

    # Create test inputs
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.ones(batch_size, num_heads, seq_len, head_dim)  # All ones for easy verification

    # Initialize the attention module
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=True)

    # Override the sliding window parameter for this test
    attn.sliding_window = 1

    # With window_size=1, each token should only attend to itself
    output = attn.eager_sliding_window_attention(
        query, key, value, attention_mask=None, window_size=1
    )

    # Check output shape
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    # Compute attention weights manually
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)

    # Create a diagonal mask
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        mask[i, i] = True
    mask = mask.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len, seq_len)

    # Apply mask
    masked_weights = attn_weights.clone()
    masked_weights = masked_weights.masked_fill(~mask, float('-inf'))
    expected_weights = torch.nn.functional.softmax(masked_weights, dim=-1)

    # Compute expected output
    expected_output = torch.matmul(expected_weights, value)

    # Compare outputs
    torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)
    print("✅ test_window_size_1 passed")


def test_window_size_larger_than_sequence():
    """Test with window_size > seq_len (equivalent to full causal attention)."""
    batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8
    hidden_size = num_heads * head_dim

    # Create a proper config
    config = BiBoConfig()
    config.hidden_size = hidden_size
    config.num_attention_heads = num_heads
    config.num_key_value_heads = num_heads  # Same as num_heads to avoid divisibility issue

    # Create test inputs
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Initialize the attention module with sliding window
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=True)

    # Create a standard causal mask (-inf for positions not allowed to attend to)
    attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len))
    attention_mask = attention_mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().unsqueeze(0).unsqueeze(0),
        float('-inf')
    )

    # With window_size > seq_len, should be equivalent to full causal attention
    window_output = attn.eager_sliding_window_attention(
        query, key, value, attention_mask=None, window_size=seq_len * 2
    )

    # For comparison, we need to implement the standard causal attention manually
    # Start with raw attention scores
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)

    # Apply causal mask (lower triangular)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    attn_weights = attn_weights + causal_mask

    # Apply softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Get expected output
    expected_output = torch.matmul(attn_weights, value)

    # Outputs should match
    torch.testing.assert_close(window_output, expected_output, rtol=1e-4, atol=1e-4)
    print("✅ test_window_size_larger_than_sequence passed")


def test_window_size_3():
    """Test with a specific window size (window_size=3)."""
    batch_size, num_heads, seq_len, head_dim = 1, 2, 6, 4  # Using 2 heads to match key_value_heads
    hidden_size = num_heads * head_dim
    window_size = 3

    # Create a proper config
    config = BiBoConfig()
    config.hidden_size = hidden_size
    config.num_attention_heads = num_heads
    config.num_key_value_heads = num_heads  # Same as num_heads to avoid divisibility issue
    config.sliding_window = window_size

    # Initialize the attention module
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=True)

    # Create inputs with a clear pattern
    # Use one-hot encodings to make the test case clearer
    query = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    key = torch.zeros(batch_size, num_heads, seq_len, head_dim)

    # Set each position to attend equally to all positions
    # (makes the attention weights uniform within the allowed window)
    for i in range(seq_len):
        for j in range(num_heads):
            for d in range(head_dim):
                query[0, j, i, d] = 1.0
                key[0, j, i, d] = 1.0

    # Create values where each position has a unique identifier
    value = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    for i in range(seq_len):
        for j in range(num_heads):
            value[0, j, i] = i + 1  # Position i has value i+1

    # Apply sliding window attention
    output = attn.eager_sliding_window_attention(
        query, key, value, attention_mask=None, window_size=window_size
    )

    # Verify each position's attention pattern
    # Each position should attend to the previous window_size-1 positions and itself
    for pos in range(seq_len):
        # Start of the window for this position (respecting sequence boundaries)
        window_start = max(0, pos - window_size + 1)
        # End of the window for this position (inclusive)
        window_end = pos

        # Expected output is the average of values in the window
        # (since query and key are the same, attention weights are uniform within the window)
        window_size_actual = window_end - window_start + 1
        expected_value = sum(i + 1 for i in range(window_start, window_end + 1)) / window_size_actual

        # Check if the output matches the expected value (for the first head)
        actual_value = output[0, 0, pos].mean().item()
        assert abs(actual_value - expected_value) < 1e-5, \
            f"Position {pos}: Expected {expected_value}, got {actual_value}"

    print("✅ test_window_size_3 passed")


def test_with_attention_mask():
    """Test interaction with an additional attention mask."""
    batch_size, num_heads, seq_len, head_dim = 1, 2, 5, 4
    hidden_size = num_heads * head_dim
    window_size = 3

    # Create a proper config
    config = BiBoConfig()
    config.hidden_size = hidden_size
    config.num_attention_heads = num_heads
    config.num_key_value_heads = num_heads  # Same as num_heads to avoid divisibility issue

    # Initialize the attention module
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=True)

    # Create inputs
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Create a custom attention mask that blocks attention to position 2
    # We'll use a very negative number (-1e9) to represent attention that's blocked
    attention_mask = torch.zeros((batch_size, 1, seq_len, seq_len))
    attention_mask[:, :, :, 2] = -1e9  # Block all attention to position 2

    # Apply sliding window attention with the additional mask
    output_with_mask = attn.eager_sliding_window_attention(
        query, key, value, attention_mask=attention_mask, window_size=window_size
    )

    # Now manually verify the attention pattern for a specific position
    # For example, position 3 should attend to positions 1,3 but not 2 (blocked by mask)
    # First compute attention scores
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)

    # Apply window mask for position 3
    window_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        for j in range(start, i + 1):
            window_mask[i, j] = True

    # Position 3 should attend to positions 1,2,3 due to window_size=3
    # But position 2 is blocked by attention_mask
    pos = 3
    valid_attention = window_mask[pos].clone()
    valid_attention[2] = False  # Position 2 is blocked

    # Get the weights after applying masks
    pos_scores = scores[0, 0, pos].clone()
    pos_scores = pos_scores.masked_fill(~valid_attention, float('-inf'))
    pos_scores = pos_scores + attention_mask[0, 0, pos]
    pos_weights = torch.nn.functional.softmax(pos_scores, dim=-1)

    # Expected output for position 3
    expected_output_pos3 = torch.matmul(
        pos_weights.unsqueeze(0),
        value[0, 0].clone()
    ).squeeze(0)

    # Verify position 3 output
    torch.testing.assert_close(
        output_with_mask[0, 0, pos],
        expected_output_pos3,
        rtol=1e-5, atol=1e-5
    )
    print("✅ test_with_attention_mask passed")


if __name__ == "__main__":
    # Run all tests
    test_window_size_1()
    test_window_size_larger_than_sequence()
    test_window_size_3()
    test_with_attention_mask()
    print("All tests passed! 🎉")
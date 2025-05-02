"""
test_model.py

Script to test BiBoModel with random input and profile memory usage and FLOPs.
"""
import os
import sys
import torch

# ensure project root in path
sys.path.append(os.path.dirname(__file__))

from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoModel

def main():
    """
    Instantiate BiBoConfig and BiBoModel, generate random input IDs,
    run model forward pass, and profile memory and FLOPs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BiBoConfig()
    model = BiBoModel(config).to(device)
    model.eval()

    batch_size = 1
    seq_length = 16
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_length), dtype=torch.long, device=device
    )

    print(f"Input IDs shape: {input_ids.shape}")

    # Memory profiling
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    print(f"Output hidden states shape: {outputs.last_hidden_state.shape}")
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak memory allocated by CUDA: {peak_memory / (1024**2):.2f} MiB")

    # FLOPs and detailed profiling (requires PyTorch >=2.0)
    try:
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                _ = model(input_ids=input_ids)
        print(prof.key_averages().table(sort_by="flops", row_limit=10))
    except ImportError:
        print("torch.profiler not available. Skipping FLOPs profiling.")

if __name__ == "__main__":
    main()

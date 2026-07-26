"""Shared fixtures for the BiBo test suite.

Everything here is deliberately TINY (hidden 64, 4 layers, vocab 512) so the whole suite runs in
well under a minute and peaks around 25 MiB of VRAM — it is a correctness suite, not a benchmark.
Runs on CUDA when available, CPU otherwise; tests that genuinely need a device are marked `gpu`.
"""
import pathlib
import sys

import pytest
import torch

# pytest inserts tests/ on sys.path, not the repo root — bootstrap it so `import src...` works
# regardless of how pytest is invoked.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configuration_bibo import BiBoConfig  # noqa: E402
from src.modeling.models import BiBoForCausalLM  # noqa: E402

# device_count() matters too: with CUDA_VISIBLE_DEVICES="" torch still reports is_available()==True
# on a CUDA build, but there is no device 0 to use.
DEVICE = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

# Small enough to be fast, big enough to exercise every branch:
#  - 4 layers with mlp_only_layers=[0, 3] -> dense at the ends, MoE in the middle
#  - 4 heads / 2 kv heads -> GQA with group size 2
#  - polyglu_mult=2 + 1 special pair -> 6 GLU experts + (+Identity) + (-Identity) = 8 routed
BASE = dict(
    vocab_size=512,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=96,
    moe_intermediate_size=96,
    polyglu_expert_multiplier=2,
    special_expert_pairs=1,
    num_experts_per_tok=2,
    max_position_embeddings=256,
    mlp_only_layers=[0, 3],
)


def make_config(**overrides) -> BiBoConfig:
    """A tiny but structurally complete BiBoConfig. Any field can be overridden."""
    return BiBoConfig(**{**BASE, **overrides})


def make_model(device=DEVICE, **overrides) -> BiBoForCausalLM:
    """A tiny BiBoForCausalLM on `device`."""
    return BiBoForCausalLM(make_config(**overrides)).to(device)


def tokens(batch=2, seq=8, device=DEVICE, vocab=BASE["vocab_size"], seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), device=device)


@pytest.fixture(scope="session")
def device():
    return DEVICE


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def model():
    return make_model()


@pytest.fixture(autouse=True)
def _free_cuda():
    """Keep peak VRAM flat across the suite on a 4 GB card."""
    yield
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def pytest_report_header(config):
    dev = DEVICE
    if DEVICE == "cuda":
        try:
            dev = f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            pass
    return f"BiBo suite — torch {torch.__version__}, device: {dev}"

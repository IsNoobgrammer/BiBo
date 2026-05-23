"""
Shared model loading and evaluation utilities for Kaggle ablation scripts.
"""
import os
import sys
import torch
import yaml

# Ensure repo root is on sys.path so `src` and `baseline` are importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM

__all__ = [
    'BASE_DIR', 'CFG', 'CFG_PATH', 'METRICS_DIR', 'PLOTS_DIR',
    'load_config', 'load_models', 'load_bibo', 'load_qwen', 'extract_routing_data',
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')


def load_config():
    """Load config.yaml and return dict."""
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


CFG = load_config()


def load_bibo(device, cfg=None):
    """Load BiBo model from checkpoint.

    Args:
        device: torch device
        cfg: config dict (defaults to CFG['bibo'])

    Returns:
        BiBoForCausalLM in eval mode
    """
    if cfg is None:
        cfg = CFG['bibo']
    bibo_cfg = {k: v for k, v in cfg.items() if k != 'device'}
    model = BiBoForCausalLM(BiBoConfig(**bibo_cfg)).to(device)
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'bibo.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  [BiBo] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"  [BiBo] WARNING: No checkpoint, using random weights")
    model.eval()
    return model


def load_qwen(device, cfg=None):
    """Load Qwen3MoE model from checkpoint.

    Args:
        device: torch device
        cfg: config dict (defaults to CFG['qwen3moe'])

    Returns:
        Qwen3MoeForCausalLM in eval mode
    """
    if cfg is None:
        cfg = CFG['qwen3moe']
    qwen_cfg = {k: v for k, v in cfg.items() if k != 'device'}
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen_cfg)).to(device)
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', 'qwen3moe.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  [Qwen3MoE] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"  [Qwen3MoE] WARNING: No checkpoint, using random weights")
    model.eval()
    return model


def load_models(device):
    """Load both models from checkpoints.

    Args:
        device: torch device

    Returns:
        (bibo_model, qwen_model) tuple
    """
    bibo_model = load_bibo(device)
    qwen_model = load_qwen(device)
    return bibo_model, qwen_model


def extract_routing_data(model, input_ids, device, model_type='bibo'):
    """Extract full routing data for a batch.

    Args:
        model: BiBo or Qwen model
        input_ids: input token ids
        device: torch device
        model_type: 'bibo' or 'qwen'

    Returns:
        dict per MoE layer with 'indices' and 'weights' tensors
    """
    model.eval()
    layer_data = {}
    hooks = []

    def make_hook(layer_idx, mtype):
        def hook_fn(_module, _inp, output):
            if mtype == 'bibo':
                indices, weights = output
                layer_data[layer_idx] = {
                    'indices': indices.detach().cpu(),
                    'weights': weights.detach().cpu().float(),
                }
            else:
                logits, scores, indices = output
                layer_data[layer_idx] = {
                    'indices': indices.detach().cpu(),
                    'weights': scores.detach().cpu().float(),
                    'logits': logits.detach().cpu().float(),
                }
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i, model_type)))

    with torch.no_grad():
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        model(input_ids=input_ids.to(device))

    for h in hooks:
        h.remove()
    return layer_data

"""Rebuild NVIDIA hub ASR models (Nemotron streaming, Parakeet TDT) into the shapes our training uses, keeping weights.

    to_hybrid(m, ctc_weight)   RNN-T / TDT model -> EncDecHybridRNNTCTCBPEModel with a fresh CTC head (NeMo's linear
                               ConvASRDecoder; train_asr's --ctc_head then swaps in our self-conditioned head)
    to_streaming(m, lookaheads, left=70)
                               a full-attention FastConformer (Parakeet) -> cache-aware streaming layout: chunked_limited
                               attention with our look-ahead presets, causal depthwise convs (subsampling unchanged). Same
                               tensors (kernel shapes do not change), different padding / masks: needs fine-tuning.
Both go through save_to -> restore_from(override_config_path=...) so the tokenizer artifacts and every matching weight
are carried over; new / reshaped parameters are reported.
"""
import os
import tempfile

import torch
from omegaconf import OmegaConf, open_dict

HYBRID = "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel"


def _rebuild(m, edit):
    import nemo.collections.asr as nemo_asr
    d = tempfile.mkdtemp(prefix="hubm_")
    src, cfgp = f"{d}/src.nemo", f"{d}/cfg.yaml"
    m.save_to(src)
    cfg = OmegaConf.create(OmegaConf.to_container(m.cfg, resolve=True))
    with open_dict(cfg):
        edit(cfg)
    OmegaConf.save(cfg, cfgp)
    old = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    n = nemo_asr.models.ASRModel.restore_from(src, override_config_path=cfgp, strict=False, map_location="cpu")
    new = n.state_dict()
    kept = sum(1 for k, v in new.items() if k in old and old[k].shape == v.shape)
    fresh = [k for k in new if k not in old]
    reshaped = [k for k in new if k in old and old[k].shape != new[k].shape]
    print(f"[hub_models] {type(m).__name__} -> {type(n).__name__}: {kept}/{len(new)} tensors carried, "
          f"{len(fresh)} new {fresh[:4]}, {len(reshaped)} reshaped {reshaped[:4]}", flush=True)
    return n


def to_hybrid(m, ctc_weight=0.3):
    def edit(cfg):
        cfg.target = HYBRID
        cfg.aux_ctc = {"ctc_loss_weight": ctc_weight, "use_cer": False, "ctc_reduction": "mean_batch",
                       "decoder": {"_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                                   "feat_in": int(cfg.encoder.d_model), "num_classes": -1, "vocabulary": []},
                       "decoding": {"strategy": "greedy"}}
    return _rebuild(m, edit)


def to_streaming(m, lookaheads, left=70):
    def edit(cfg):
        e = cfg.encoder
        e.att_context_style = "chunked_limited"
        e.att_context_size = [[-1, -1] if r < 0 else [left - left % (r + 1), r] for r in lookaheads]
        e.att_context_probs = None
        e.conv_context_size = "causal"
        # causal_downsampling is NOT switched: it changes pre_encode.out's input size (Parakeet 4096 -> 4352), so its
        # trained weights would not load; the stock subsampling's few frames of look-ahead are kept instead
    return _rebuild(m, edit)


def clean_none(dl_cfg):
    """Hub configs carry keys set to None that newer NeMo's LhotseDataLoadingConfig types as int (e.g.
    bucket_buffer_size): drop them so the dataclass defaults apply."""
    with open_dict(dl_cfg):
        for k in [k for k, v in dl_cfg.items() if v is None]:
            dl_cfg.pop(k)
    return dl_cfg

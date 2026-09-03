"""Ablation trainer — one arm, one seed, one run (~6h on RTX 6000). W&B for graphs/logs.

  python -m ablate.common.train --arm bibo_min --seed 0 --tokens 1_000_000_000 \
      --batch 40 --seq_len 1024 --precision bf16 --patches liger_norm,liger_rope,ce,moe --wandb

Local smoke:  --data synthetic --max_steps 5 --batch 2 --seq_len 128  (no --wandb)
No seed aggregation / variance machinery by design: pass --seed, read W&B; compare seeds there.
"""
from . import _paths  # noqa: F401
import os
import json
import time
import math
import argparse
import contextlib
import torch
from .models import build_arm, count_params
from .configs import SHARED, glu_count, swa_block_pattern, hswa_windows, resolve_swa
from . import patches as patchmod
from .optim import build_optimizers
from .schedule import make_scheduler, make_wd_schedule
from .data import token_batches, TRAIN_DATASET
# EVAL REMOVED Aug 7 2026. The whole ablate/common/eval package and evaluate.py are gone; runs log
# train loss, grad norm and throughput only. Nothing here is comparable to any previous board.
# Kept because the HF checkpoint push still needs a tokenizer to save alongside the weights.
TOKENIZER = "fhai50032/QTK-81K"
from .router_trace import RouterTrace   # training-stream router diagnostic; survived the purge
from . import validation as _val        # frozen small batch, CE logged on the training log line
from .tensor_health import tensor_norms
from .per_layer import PerLayerRouter, per_layer_params
from kernels.sm120.cross_entropy import fused_linear_cross_entropy   # sm120 (Blackwell); CE byte-identical to sm75

DEV = "cuda"
_DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
# THE ACTIVATION IS RADIAL NormSiLU, and as of Aug 1 2026 it is the ONLY one src implements --
# --act, --act_tail and the whole act-cycle plumbing are gone with it. Decided at 1B tokens, bpb vs
# a 0.00037 same-seed floor: radial 0.64313 < silu-a 0.64429 < normsilu 0.64646 < silu 0.64768.
# radial wins because its magnitude control is BOUNDED: p = sigmoid(theta) in (0,1) makes p->0
# exactly normsilu and p->1 full magnitude, and the learned p is a DEPTH RAMP 0.11 -> 0.93, so a
# layer runs as normsilu early and full-magnitude late. Kernel act code 8; theta lives on
# BiBoFusedExperts.radial_theta and trains through AdamW (--act_scale_lr gives it its own rate).
RADIAL_ACT_CODE = 8


class _QwenAuxCollector:
    """Hooks Qwen's routers to grab per-layer gate logits (the vendored Qwen3MoeModel doesn't return them),
    so we can add the Switch-style load-balancing aux loss ourselves (Qwen's native balancing = fair vs
    BiBo's bias balancing). No-op for BiBo."""
    def __init__(self, model):
        self.logits = []
        self._handles = [m.register_forward_hook(self._hook)
                         for _, m in model.named_modules() if m.__class__.__name__ == "Qwen3MoeTopKRouter"]

    def _hook(self, module, inp, out):
        self.logits.append(out[0])           # (num_tokens, num_experts)

    def reset(self):
        self.logits = []


def _ce(model, ids, use_fused, aux=None, aux_coef=0.0, num_experts=6, top_k=2, pad_id=None):
    """CE over next-token targets, with `pad_id` (from --pad_id) excluded from the loss.

    Default None = nothing is masked, which is correct for a PACKED corpus: it contains no padding,
    so masking any id would only delete real tokens. Set --pad_id only for a corpus that pads rows.

    Both branches take ignore_index explicitly. They used to take neither, so the kernel's default
    of -100 applied and a padded corpus would have trained ON its padding -- pad ids are
    non-negative and never matched -100.

    In QTK-81K pad_token_id == eos_token_id == 0 (<|endoftext|>), so masking id 0 also masks every
    end-of-text target. That is safe ONLY while documents are delimited by something else --
    <|im_end|> (81914) in this corpus. If a future corpus separates documents with <|endoftext|>,
    masking it would stop the model ever learning to terminate. main() warns on this.
    """
    inp, tgt = ids[:, :-1], ids[:, 1:].reshape(-1)
    if aux is not None:
        aux.reset()
    out = model.model(input_ids=inp, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    sh = h.reshape(-1, h.shape[-1])
    _ign = -100 if pad_id is None else int(pad_id)
    loss = (fused_linear_cross_entropy(sh, model.lm_head.weight, tgt, ignore_index=_ign) if use_fused
            else torch.nn.functional.cross_entropy(model.lm_head(sh), tgt, ignore_index=_ign))
    if aux is not None and aux_coef > 0 and aux.logits:      # Qwen aux load-balancing loss
        from baseline.qwen3moe.modeling import load_balancing_loss_func
        loss = loss + aux_coef * load_balancing_loss_func(tuple(aux.logits), num_experts, top_k)
    return loss


def _measure_peak_tflops(device, dtype, n=8192, iters=30):
    """Self-calibrating MFU denominator: achievable dense matmul TFLOPS on THIS gpu/dtype."""
    if device != "cuda":
        return 0.0
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    return (2 * n ** 3 * iters) / (time.time() - t) / 1e12


def _bool(v):
    """argparse bool that still accepts the repo's `--flag 1` / `--flag 0` convention."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@torch.no_grad()
def _expert_corr(model):
    """Mean cross-expert off-diagonal |cosine| over the 3D MoE expert stacks (0 = orthogonal experts,
    1 = identical). Diagnostic for xorth: does whitening actually decorrelate the experts over training?"""
    vals = []
    for n, p in model.named_parameters():
        if p.ndim == 3 and ("expert" in n or "gate_up_proj" in n or "down_proj" in n) and p.shape[0] > 1:
            e = p.shape[0]
            x = p.detach().reshape(e, -1).float()
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            m = x @ x.t()
            vals.append((m - torch.diag(torch.diagonal(m))).abs().sum().item() / (e * e - e))
    return sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def _router_corr(model):
    """Mean off-diagonal |cosine| between the ROUTER's per-expert weight rows (0 = the experts are
    scored along orthogonal directions, 1 = the router has collapsed onto one direction and can no
    longer tell experts apart).

    NOT the same thing as `_expert_corr`, which walks the 3D MoE expert STACKS. This walks the 2D
    router projection -- `.gate.gate_proj.weight` (E,H) or `.gate.gate_conv` (E, H*K) -- and it is
    the metric the conv-router axis fix is about: an nn.Conv1d (E,H,K) weight sends Muon's NS to a
    (K,K) gram, which decorrelates kernel TAPS and lets THIS number climb. Storing the weight 2D
    puts the gram back at (E,E). If conv is working as intended, rcorr must track the MLP router's,
    not run away from it."""
    vals = []
    for n, p in model.named_parameters():
        if ".gate.gate_proj.weight" in n or ".gate.gate_conv" in n:
            x = p.detach().reshape(p.shape[0], -1).float()
            e = x.shape[0]
            if e < 2:
                continue
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            m = x @ x.t()
            vals.append((m - torch.diag(torch.diagonal(m))).abs().sum().item() / (e * e - e))
    return sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def _typed_memory_stats(model):
    """Learned depth timescales and innovation strength for typed-memory runs."""
    fast, slow, innovation, controller_rms = [], [], [], []
    result = {}
    for module_name, module in model.named_modules():
        prefix = f"train/typed/{module_name.removeprefix('model.').replace('layers.', 'layer_')}"
        for name, value in (getattr(module, "_typed_diagnostics", None) or {}).items():
            result[f"{prefix}/{name}"] = value.item()
        # Disable capture before validation; only detached training-forward scalars are kept.
        if getattr(module, "use_typed_attn_res", False):
            module._typed_diagnostics = None
        for name, value in module.named_parameters(recurse=False):
            if "_res_type_controller" in name:
                result[f"{prefix}/{name}_rms"] = value.detach().float().square().mean().sqrt().item()
            elif "_res_type_bias" in name:
                for type_id, bias in enumerate(value.detach().float()):
                    result[f"{prefix}/{name}_{type_id}"] = bias.item()
        if hasattr(module, "typed_attn_res_fast_decay_logit"):
            f = module.typed_attn_res_fast_decay_logit.detach().float().sigmoid()
            gap = module.typed_attn_res_slow_decay_gap_logit.detach().float().sigmoid()
            fast.append(f)
            slow.append(f + (1.0 - f) * gap)
            result[f"{prefix}/fast_decay"] = f.item()
            result[f"{prefix}/slow_decay"] = slow[-1].item()
            controller_rms.append(
                module.typed_attn_res_slow_write_controller.detach().float().square().mean().sqrt()
            )
        if hasattr(module, "typed_attn_res_innovation_logit"):
            innovation.append(
                module.typed_attn_res_innovation_logit.detach().float().sigmoid()
            )
            result[f"{prefix}/innovation_alpha"] = innovation[-1].item()
    for name, values in (
        ("typed_fast_decay", fast),
        ("typed_slow_decay", slow),
        ("typed_innovation_alpha", innovation),
        ("typed_slow_write_controller_rms", controller_rms),
    ):
        if values:
            stacked = torch.stack(values)
            result[f"train/{name}_mean"] = stacked.mean().item()
            result[f"train/{name}_min"] = stacked.min().item()
            result[f"train/{name}_max"] = stacked.max().item()
    return result


def _save_hf_ckpt(model, tokenizer, out_dir):
    """Write a reload-ready bf16 HF checkpoint (config.json + safetensors + tokenizer) to out_dir. Runs on
    the MAIN thread between steps (fast). Casts only the big matrices (ndim>=2: linears, embeddings, expert
    stacks) to bf16 in a fresh state-dict COPY — the live fp32 master weights are untouched (casting them in
    place would corrupt training). Keeps 1D params (RMSNorm/LayerNorm gains, biases) and all buffers (RoPE
    inv_freq, router bias) at fp32: they're precision-sensitive and tiny, so bf16 buys no size but loses
    precision. Unwraps torch.compile so the state-dict keys are clean (compiled model.model has `_orig_mod.`
    prefixes that would make the checkpoint unloadable)."""
    os.makedirs(out_dir, exist_ok=True)
    compiled = getattr(model, "model", None)
    orig = getattr(compiled, "_orig_mod", None)   # set iff model.model was torch.compile'd
    if orig is not None:
        model.model = orig
    prev_dtype = getattr(model.config, "torch_dtype", None)
    try:
        _params = set(n for n, _ in model.named_parameters())        # cast matrices only; not norms/biases/buffers
        sd = {k: (v.to(torch.bfloat16) if (k in _params and v.is_floating_point() and v.ndim >= 2) else v)
              for k, v in model.state_dict().items()}
        model.config.torch_dtype = torch.bfloat16                    # so from_pretrained loads as bf16
        model.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
    finally:
        model.config.torch_dtype = prev_dtype
        if orig is not None:
            model.model = compiled
    tokenizer.save_pretrained(out_dir)
    return out_dir


def _push_hf_async(api, repo, local_dir, path_in_repo, tag):
    """Fire-and-forget upload (rayon-style): upload_folder runs in a background thread (run_as_future), so
    training continues immediately. On completion the local dir is reclaimed. Returns the Future to drain
    at process exit — a detached nohup process would otherwise kill the upload thread on exit."""
    fut = api.upload_folder(folder_path=local_dir, path_in_repo=path_in_repo, repo_id=repo,
                            commit_message=f"checkpoint {tag}", run_as_future=True)

    def _done(f, _d=local_dir, _t=tag):
        try:
            f.result()
            print(f"  [hf] pushed {_t} -> done", flush=True)
        except Exception as e:
            print(f"  [hf] push {_t} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
        finally:
            import shutil
            shutil.rmtree(_d, ignore_errors=True)   # reclaim disk once uploaded
    fut.add_done_callback(_done)
    return fut


# _eager() lived here: it swapped the compiled module for the eager one around eval/generation,
# because torch.compile choked on the eval shapes. Deleted Aug 7 2026 with the eval -- its only
# callers were the eval and sampling call sites. Restore from git history if sampling comes back.


def _interp(rt):
    """Move every diagnostic out of `train/` and into `interp/`.

    `train/` is the panel you read while a run is in flight -- loss, lr, mfu, tps, grad norm. The
    per-layer router/XSA/carry/radial tables are 200+ series and they were burying those five.
    They keep their names, only the section changes, so `train/attn_res_s/L0` from an older run and
    `interp/attn_res_s/L0` from a new one are the same series under a different heading.

    Keys that are not already `train/*` (params/norm, grad/norm) are left alone: a dead gradient is
    a training failure, not an interpretability question, and it belongs where you will see it.
    """
    return {("interp/" + k[6:] if k.startswith("train/") else k): v for k, v in rt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["qwen", "bibo_min"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=1_000_000_000)
    # 64x2 measured Jul 28 2026 at 64 experts/k=8: 156.3k tps vs 147.3k for 32x4, a free 6.1% at the
    # SAME 128 seq/step (same update count, same math up to accumulation rounding, so results stay
    # comparable). bs=64 costs 73.7 G of 96; bs=96 would not fit. ga is memory-free.
    ap.add_argument("--batch", type=int, default=64)             # per micro-step
    ap.add_argument("--grad_accum", type=int, default=2)          # global batch = batch * grad_accum
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")  # NEVER fp16
    ap.add_argument("--attn", choices=["sdpa", "flash_attention_4"], default="sdpa")
    # Token id excluded from the CE. None = nothing masked, correct for a PACKED corpus. Set it
    # only when rows are actually padded. See _ce for the pad/eos collision this has to respect.
    # Free-text suffix on the W&B run name. The auto-name is built from ARCHITECTURE flags only,
    # so a sweep over a non-architecture knob (lr, wd, seed) produces N runs with ONE name and is
    # unreadable in the UI. Use this to tag the swept value, e.g. --run_tag mlr1e-2.
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--pad_id", type=int, default=None)
    ap.add_argument("--aux_coef", type=float, default=0.001)                      # Qwen aux load-balancing loss coef (0=off; paper 0.001)
    ap.add_argument("--experts", type=int, default=0)   # TOTAL routed experts (GLU + specials); 0 = SHARED (6). GLU count = experts - 2*special_pairs -- see configs.glu_count
    # PolyGLU expert activation. THE ACTIVATION AXIS IS CLOSED (Jul 26 2026): the six on/off switches
    # it was ablated with are gone and this is a single choice, DEFAULT silu. The other codes stay in
    # the kernel and stay selectable -- pass a comma list to revive a MIXED cycle (e.g. --act silu,situ
    # -> 0,5,0,5,...), which is how a future NormSiLU+NormSiTU mixture would be run.
    #   silu 0 (default) | relu2 1 | normsilu 2 | situ 5 | normrelu2 6 | normsitu 7
    # Measured @e30 500M tok: acts-n (normsilu) 0.7273, acts-s (silu) 0.7444, Z (normrelu2) 0.8345.
    # Per-expert INPUT SCALE alpha for ANY activation: act(alpha_e * x), x = gate (silu/relu2) or
    # gate/rms(gate) (the normed codes; alpha sits AFTER the rms or it is exactly inert). Reuses the
    # situ (alpha,gamma) params -- gamma gets ZERO gradient for non-SiTU codes and stays 1.0, since a
    # per-expert OUTPUT gain is redundant with the router weight (g_e*w*f == (g_e*w)*f). 1D (E,) ->
    # AdamW by the ndim rule. Tag _aS.
    # alpha must travel from 1 to ~5 to matter; AdamW moves a param ~lr/step, cosine-averaged ~0.5*lr,
    # so reachable travel over N steps is ~0.5*lr*N. At adam_lr 5e-4 / 2000 steps that is 0.5 -- alpha
    # cannot get there and the arm reads as a null for the wrong reason. 1e-2 gives travel ~10.
    # DEFAULT 0.01, not adam_lr. theta is an exponent LOGIT that has to TRAVEL: the measured depth
    # ramp is p 0.11 -> 0.93, i.e. theta from -2.08 to +2.56, a span of 4.6 from an init of 0.
    # AdamW's per-step displacement is ~lr (the update is sqrt(v)-normalized to O(1)), so at
    # adam_lr=5e-4 over 4000 steps the ENTIRE travel budget is 2.0 -- under the 2.56 the deep layers
    # need, and that assumes a perfectly sign-consistent gradient. Every radial result on the board
    # was produced at 0.01 (20x adam_lr), where the budget is 40 and the constraint is not binding.
    # Sharing adam_lr would leave p stuck near its 0.5 init and look like "the axis does nothing".
    # 0 = share --adam_lr (the old default; kept so the A/B is one flag).
    ap.add_argument("--act_scale_lr", type=float, default=0.01)
    # Route (1,H) "matrices" -- AttnRes pseudo-queries are nn.Linear(hidden,1) -- to AdamW instead
    # of Muon. Default False = the ndim rule as it stands, which is what every arm so far ran.
    ap.add_argument("--vec_matrices_adamw", type=_bool, default=False)
    # ...and WHICH AdamW group they land in. "default" = lr adam_lr, wd wd (step 0.0027 measured);
    # "act" = the act-scale group, lr act_scale_lr, wd 0 (step 0.054). Muon's is 0.0449, so the two
    # settings bracket it. Only meaningful with --vec_matrices_adamw true.
    ap.add_argument("--vec_adamw_group", choices=("default", "act"), default="default")
    # radial p parameterization. sigmoid = every result on the board; tanh additionally lets
    # p go NEGATIVE (gain r^p < 1, shrinking high-rms rows), which sigmoid cannot express.
    # Kernel act code 8 vs 10. Tag _ptanh.
    ap.add_argument("--radial_p", choices=["sigmoid", "tanh"], default="sigmoid")
    # Expert activation. "radial" = r^p*SiLU(g/r), the adopted default. "silu" = plain SwiGLU
    # (kernel act code 0), the arm that asks what the activation is still worth inside the current
    # SWA+XSA stack. Kernel-path only -- src eager has been radial-only since the Aug 1 debloat.
    ap.add_argument("--act", choices=["radial", "silu"], default="radial")
    # Kimi K3 Attention Residuals (exp/). "off" = stable src model. "control" = exp's model with
    # residuals disabled. An INT is the block size in decoder layers: 1 = per-layer (Full AttnRes),
    # 3 = one block per [G,S,S]. At 10 layers K3's own default of 12 is degenerate (one block).
    ap.add_argument("--attn_res", default="off")
    # 2 = K3 faithful (a depth-mix before BOTH the attention and the MLP sublayer). 1 = one mix
    # per layer at the layer input; the MLP takes an ordinary PreNorm residual. Halves the
    # depth-attention work and the AttnRes parameter count.
    ap.add_argument("--attn_res_sites", type=int, choices=[1, 2], default=2)
    # sites=1 only: MLP reads (site-1 mix + attn_output) instead of the raw prefix sum.
    ap.add_argument("--attn_res_carry", type=_bool, default=False)
    # Typed thought/memory extension (exp/ only): keep attention and MLP contributions as
    # distinct candidates and add a token-conditioned type score to each AttnRes read.
    ap.add_argument("--typed_attn_res", action="store_true")
    # Archive one memory-only state per completed block in addition to the canonical block state.
    ap.add_argument("--typed_attn_res_long_memory", type=_bool, default=True)
    # Relative initial prior of each extra typed candidate versus a canonical K3 candidate.
    ap.add_argument("--typed_attn_res_extra_init", type=float, default=0.01)
    # Two depth timescales: fast memory resets at block boundaries; slow memory persists.
    ap.add_argument("--typed_attn_res_fast_slow_memory", action="store_true")
    ap.add_argument("--typed_attn_res_fast_decay_init", type=float, default=0.5)
    ap.add_argument("--typed_attn_res_slow_decay_init", type=float, default=0.95)
    # Store the part of each MLP output not already parallel to the current thought stream.
    ap.add_argument("--typed_attn_res_innovation_write", action="store_true")
    ap.add_argument("--typed_attn_res_innovation_init", type=float, default=0.01)
    # Keep the residual stream in the layer-input dtype (fp32 under autocast, as the
    # standard-residual control does) so AttnRes is the ONLY difference from the baseline.
    ap.add_argument("--attn_res_fp32_stream", type=_bool, default=False)
    # Learnable per-layer coefficient on the carry term: A_coeff = 2*sigmoid(theta), init 1.0
    # so it is a strict generalization of plain carry. Logged as cs= to get the depth profile.
    # none = c fixed at 1 (no parameter). raw = c learnable, unconstrained. rms = c learnable AND
    # the stream it multiplies is RMS-normalised, which is what removes the need to bound c: the
    # bounded transforms existed only to stop an unbounded c running away (7936 by step 400).
    ap.add_argument("--attn_res_carry_scale",
                    choices=["none", "raw", "unbounded", "rms", "sigmoid", "tanh",
                             "sigmoid_rms", "tanh_rms", "slope05", "slope15"], default="none")
    # Third, OFF-SIMPLEX term on the carry write: h = attn_read + c*attn_out + d*embedding.
    # d is per-layer, learnable, init 0 -> strict generalization of plain carry. Tests whether
    # layer 1's negative XSA alpha is a workaround for a missing token-identity channel.
    # Force the EAGER carry write even when the fused kernel imports. The kernel is provably
    # correct (parity + a per-call probe at 2.5e-03, exactly the bf16 scalar rounding it removes)
    # but it perturbs MoE top-k for ~3.9% of token-layer decisions from step 0, so a kernel run is
    # a different TRAJECTORY, not the same model. This flag exists to run matched same-box pairs
    # and find out whether the observed +0.0037 bpb is a real cost or seed noise.
    ap.add_argument("--fused_res_add", type=_bool, default=True)
    ap.add_argument("--attn_res_emb_term", type=_bool, default=False)
    # d = f(theta). "none" = raw (what the first emb arm ran). Plain "sigmoid" caps d at 1.0,
    # which CLIPS layer 1 -- it asked for 1.32, the largest value on that axis. "2sigmoid"
    # spans (0,2) and covers it.
    ap.add_argument("--attn_res_emb_scale",
                    choices=["none", "sigmoid", "2sigmoid", "tanh", "2tanh"], default="none")
    # WHERE the embedding skip is applied. "mlp" = the retired path, d*emb on the carry write.
    # "ht" = a RADIAL gain on the depth-mix output: emb * r^(p-1), r = rms(emb) per token,
    # p = sigmoid(theta) in (0,1), theta init -4 so it starts near UNIT NORM. At "ht" the skip
    # also reaches ATTENTION, since input_layernorm(HT) feeds self_attn -- the mlp site never did.
    ap.add_argument("--attn_res_emb_site", choices=["mlp", "ht"], default="mlp")
    # REPLACE the radial exponent with a flat gain on the RAW embedding: HT = AR(...) + i * emb.
    # One scalar per layer, no norm of any kind, theta is not created, and the add goes through
    # the fused residual kernel as a second stream. Both normed ht arms lost (radial 0.6672 vs
    # 0.6579 c-only) while the only emb arm that ever won -- mlp-site d*emb, 0.6572 -- was
    # unnormed, and the MoE-output-norm round measured "norm the addend" at -0.010 bpb in all 3
    # variants. i inits 0 (strict generalization). ht site only.
    ap.add_argument("--attn_res_emb_gain", type=_bool, default=False)
    # Depth-score -> weight map. BOTH are convex combinations (weights sum to 1), so this does
    # not change whether the read is a weighted average -- only how scores land on the simplex.
    # softmax is shift-invariant (N-1 usable dof); signorm = sigmoid(x_i)/sum_j sigmoid(x_j) is
    # shift-sensitive (all N live) and saturates toward uniform at large positive scores.
    ap.add_argument("--attn_res_score", choices=["softmax", "signorm"], default="softmax")
    # Sparse depth mix: keep only the k highest-scoring candidates, prefix_sum always among
    # them, renormalize over the survivors. 0 = dense. Inert wherever the pool is <= k, so at
    # --attn_res 1 it first binds at layer k-1 and at --attn_res 3 (pool never exceeds 5) any
    # k >= 5 does nothing at all.
    ap.add_argument("--attn_res_topk", type=int, default=0)
    # c per HIDDEN CHANNEL instead of one scalar per layer: M_in = attn_read + c (*) attn_out.
    # 512 params/layer. Real capacity, not a reparameterization -- attn_output also feeds the
    # prefix-sum stream, so a diagonal scale on the MLP-facing copy cannot fold into o_proj.
    ap.add_argument("--attn_res_carry_per_dim", type=_bool, default=False)
    # c = SiLU(W @ attn_read), input-dependent, bias-free. Subsumes per_dim and carry_scale.
    # W is (hidden, hidden) per layer -- at H=512 that is 2.6M params (+1.9%); at the 7168 both
    # Kimi K3 and DeepSeek-V3 use it would be 51.4M/layer, i.e. one extra attention projection.
    ap.add_argument("--attn_res_carry_gate", choices=["none", "diag", "full"], default="none")
    # d per hidden channel, the widening that made c worth -0.0011 for free.
    ap.add_argument("--attn_res_emb_per_dim", type=_bool, default=False)
    # BF16 RESIDUAL STREAM (src/ baseline only). The stream is fp32 by default -- not by
    # choice, but because weights are fp32 master and nn.Embedding is not autocast, so
    # inputs_embeds is fp32 and every residual add promotes back up. modded-nanogpt runs
    # the whole stream in bf16. Master weights and optimizer state are unaffected.
    ap.add_argument("--bf16_residual_stream", type=_bool, default=False)
    # Round the sublayer OUTPUT to bf16 before the residual add, stream stays fp32.
    ap.add_argument("--bf16_moe_out", type=_bool, default=False)
    # init for the (E,) act-scale param. 1.0 for the INPUT-SCALE codes (alpha multiplies the gate, so
    # 1 = feature off). 0.0 for radial (code 8), where the param is the exponent LOGIT and
    # p=sigmoid(0)=0.5 is the intended start -- leaving it at 1.0 would silently start p at 0.731.
    # gamma init, code 9 only (gamma is inert elsewhere). Needed because gamma*SiLU(alpha*g) is
    # DEGENERATE once |alpha*g| is large -- it collapses to gamma*alpha*g, so only the product
    # matters and AdamW moves both params together. Starting alpha near 1/rms(gate) and gamma near
    # rms(gate)^p puts the gate in its curved region from step 0, where the two grads differ.
    # PER-LAYER activation: MoE layers >= --act_tail_from use --act_tail, earlier layers use --act.
    # Motivated by measurement: CV(rms(gate)) across tokens is 28% at MoE layer 0 but only ~4% from
    # layer 2 on, so NormSiLU's per-token norm earns its keep early and is nearly a constant divide
    # later -- where a per-expert scalar reproduces it and skips the RMS pre-pass.
    ap.add_argument("--special_pairs", type=int, default=0)
    # ASYMMETRIC identity counts. --special_pairs is per-type and applies to BOTH signs, so it
    # cannot express e.g. 6 +Identity with 2 -Identity. These override it independently.
    ap.add_argument("--pos_identity_n", type=int, default=None)
    ap.add_argument("--neg_identity_n", type=int, default=None)                       # BiBo param-free special experts, per-type count
    ap.add_argument("--no_pos_identity", dest="pos_identity_expert", action="store_false")  # drop +Identity (code 3); test -Identity alone
    ap.add_argument("--no_neg_identity", dest="neg_identity_expert", action="store_false")  # drop -Identity (code 4); test +Identity alone
    ap.add_argument("--norm_topk_prob", type=int, default=1)  # 1 = normalize top-k weights to sum to 1 (BiBo model default). 0 = raw scores as weights (old ablate behavior)
    ap.add_argument("--router_log", type=int, default=1)
    ap.add_argument("--router_optim", choices=["muon","adamw"], default="muon")  # router proj optimizer; muon = current default (2D -> Muon by the ndim rule); tag _radamw      # per-step router mechanics (GPU-accumulated, 1 sync per log_every)
    # MoE-branch magnitude knobs (applied AFTER the top-k norm; normalization sets the SPLIT, these set LOUDNESS)
    # SWA. "block3" = [G,S,S] x N + a global tail (configs.swa_block_pattern); "none" = all global;
    # DEFAULT STACK as of Aug 3 2026: [G,S,S] blocks + a global tail, w128, XSA learnable on every
    # layer, radial experts. That is the configuration the 524M board settled on, so it is now what
    # you get without flags; "none" is the all-global ablation.
    ap.add_argument("--swa_pattern", default="block3")
    # int = uniform. Comma list = HIERARCHICAL: the cycle applied to the windowed layers within
    # each block, e.g. "128,512" -> first windowed layer of every block 128, second 512.
    ap.add_argument("--sliding_window", default="128")
    # QK-norm on the WINDOWED layers only (global layers always keep it). False = the arm that
    # asks whether a 128-token span already bounds logits well enough to make it redundant.
    ap.add_argument("--swa_qk_norm", type=_bool, default=True)
    # Positional encoding is FIXED: NoPE on full-attention layers, full RoPE on windowed ones.
    # Not a knob -- the rope round settled it. This is only the base for the windowed layers.
    ap.add_argument("--rope_theta", type=float, default=None)
    # Which layers keep a DENSE FFN instead of the MoE block. Default [0, 9] on the board config,
    # which under swa_pattern block3 are exactly the full-attention layers. "none" = every layer
    # is MoE. Changing this changes the PARAMETER COUNT -- these arms are not param-matched, and
    # the comparison is quality per (total, active) params, both of which train.py prints.
    ap.add_argument("--mlp_only_layers", type=str, default=None,
                    help='comma list of dense-FFN layer indices, or "none" for all-MoE')
    # Width of the DENSE FFN on mlp_only_layers. Default 1024 leaves a dense layer far cheaper
    # than the MoE block it sits next to, so swapping one for the other moves active params by
    # +5.0% and the arms are not comparable at matched compute. top_k*moe_inter (=4608 on the
    # board config) makes a dense layer and an MoE layer cost the SAME active params, to within
    # the router's h*num_experts.
    ap.add_argument("--dense_inter", type=int, default=None)
    # PER-LAYER expert geometry, "layer:E:top_k:width" (repeat with commas). Holding both active
    # and total params fixed pins E/top_k = total/active, so a COARSER layer needs a smaller k:
    # on the board config 0:32:3:1536 costs the same as the default 64/6/768 to 0.2%, while
    # 0:32:6:768 would be half the layer. The point is to give the entry layer a decision it can
    # actually make -- its router ends training unable to separate rank 6 from rank 7 (boundary
    # gap 0.0026 vs 0.02-0.10 elsewhere) -- without paying the 0.10 ctx4095 that a dense L0 costs.
    ap.add_argument("--moe_override", default="",
                    help='per-layer expert geometry, e.g. "0:32:3:1536"')
    # XSA is part of the default stack now (learnable per-head alpha, init 0). BooleanOptionalAction
    # so the existing `--use_xsa` spelling still parses and `--no-use_xsa` is the ablation.
    ap.add_argument("--use_xsa", action=argparse.BooleanOptionalAction, default=True)
    # Per-head rejection-strength LOGIT; strength = tanh(init). Ships with --use_xsa, it is not a
    # separate axis: learnable-alpha XSA beat fixed full-strength XSA by 20x the noise floor at
    # matched step 1000, so full strength (init inf) is not on the menu. 0 = XSA starts OFF and the
    # model has to switch it on, which is what that run did. Tag _xaI<v> when moved off 0.
    ap.add_argument("--xsa_alpha_init", type=float, default=0.0)
    ap.add_argument("--n_shared", type=int, default=0)   # shared experts as a WIDTH multiple of moe_intermediate_size (Kimi K3 form). 0 = no shared expert. Added UNSCALED to the routed sum. Tag _sh<N>
    # Per-token norm on the MoE BLOCK OUTPUT (combined expert sum) just before the residual add.
    # "rms" keeps a learnable per-channel gain; "unit" is gain-free so ONLY the mixture direction
    # survives. Pairs with --norm_topk_prob 0: let the router weights run unbounded, then pin the
    # branch magnitude here instead. Tag _mon-<v>
    ap.add_argument("--top_k", type=int, default=0)                # 0 = SHARED (2). Raising it WITHOUT --moe_inter multiplies active expert FLOPs by the same factor. Tag _k<n>
    ap.add_argument("--moe_inter", type=int, default=0)            # 0 = SHARED (768). Halve it when doubling top_k to hold compute constant. Tag _mi<n>
    ap.add_argument("--bias_update_threshold", type=int, default=10240)           # tokens between bias updates (if bias)
    ap.add_argument("--bias_update_factor", type=float, default=-1.0)             # <0 = config default, which is MODE-DEPENDENT (prop 0.4, sign 0.001) because u means different things; 0 = balancing off
    ap.add_argument("--compile", action="store_true")           # torch.compile the transformer body
    ap.add_argument("--peak_tflops", type=float, default=0.0)   # MFU denominator: 0=auto-measure achievable GEMM;
    #                                                             else theoretical, e.g. 480 (dense bf16) / 960 (sparse)
    # 'megakernel' is DEFAULT: it fuses post_attention_layernorm + router + experts into one block
    # and supersedes 'moe' on MoE layers. 'moe' stays in the list because the dense mlp_only_layers
    # and the Qwen arm still route through BiBoFusedExperts.forward.
    ap.add_argument("--patches", default="liger_norm,liger_rope,ce,moe,megakernel")
    ap.add_argument("--muon_scale_mode", choices=["polar", "normuon", "aurora", "aurora_ema", "aurora_ema_v2"],
                    default="aurora")  # post-NS row scaling; EMA variants: normuon / aurora_ema / aurora_ema_v2
    ap.add_argument("--xorth_post", type=float, default=0.0)       # cross-expert whitening MAX strength (0=off), scoped to MoE expert stacks
    ap.add_argument("--xorth_gate_ref", type=float, default=0.3)   # correlation gate: full whitening at off-diag RMS>=this; below it ramps to ~0; <=0 disables gate
    ap.add_argument("--xorth_ema", type=float, default=0.95)       # EMA decay of the persistent per-stack (E,E) gram
    ap.add_argument("--xorth_warmup_steps", type=int, default=0)   # gate xorth OFF until step > this (0 = active from step 1)
    ap.add_argument("--xorth_where", choices=["pre", "post"], default="post")  # whiten momentum PRE-NS or orthogonalized update POST-NS
    # 1e-2 is the BOARD LR. It used to be 3e-4 here while every launch script passed --muon_lr 0.01,
    # so the default was dead code that read as the real setting -- an argparse default nobody runs
    # is worse than no default. Anything not passing --muon_lr now gets what the board runs.
    ap.add_argument("--muon_lr", type=float, default=1e-2)
    ap.add_argument("--adam_lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    # REVERSE-COSINE wd ramp: --wd is the START, --wd_end the finish, rising while the LR decays.
    # wd 0.01 leads on train loss through ~step 1500 then loses ~0.021 in the anneal (BOTH silu and
    # normsilu, 0.0207 / 0.0215) because the relative step sqrt(2*lr*wd) collapses as lr -> 0.
    # Muon-only cautious decay: skip the decay on coordinates the update is already shrinking.
    # WARNING on comparability: every baseline on the board (normsilu 0.67726, silu 0.68171, aS-s
    # 0.67674, rad-n 0.67669) was trained NON-cautious, so any cautious run must be A/B'd against
    # them rather than against another cautious run. Runs are tagged _cwd when this is on.
    ap.add_argument("--cautious_decay", type=_bool, default=False)   # OFF by default: the whole board is non-cautious, so a default-on run is not comparable to it
    # ---- OPTIMIZER AXIS: muon (every baseline on the board) | manas (muon + rolling probe) ----
    # Manas runs the SAME sm120 gram-NS Muon step and takes every Muon argument verbatim; it adds a
    # lookahead probe -- forward/backward evaluate at theta + gamma*D, where D is a long-memory
    # consensus of UNIT-normalized micro-batch gradient directions (one vote per grad_accum micro).
    # It is a GRADIENT-ACCUMULATION optimizer: at 1 vote/step it self-gates to plain Muon (measured
    # wall-clock negative there); the edge grows with votes (-0.046 @ 4v, -0.140 @ 8v vs LR-TUNED
    # muon on the legacy 137M harness, 300-step train loss). --grad_accum IS the vote count.
    # Everything measured to date is 300-step TRAIN loss on a different harness, and train loss is
    # read AT THE PROBED THETA (downhill), so the bpb eval here is the first honest verdict.
    ap.add_argument("--optim", choices=["muon", "manas"], default="muon")   # tag _manas
    # probe dose. 0 = AUTO from the measured law gamma = 0.08*sqrt(lr/3e-4)*k/sqrt(m), k=grad_accum,
    # m=--batch (validated 6/6 blind across 32/64/128 x ga1/2/4). Set explicitly to override.
    # gamma is a function of LR -- if you change --muon_lr, the auto value tracks it.
    ap.add_argument("--probe_gamma", type=float, default=0.0)
    ap.add_argument("--probe_rho_step", type=float, default=0.96)  # consensus memory in STEPS (1/(1-rho)); flat 0.96-0.99
    # 0 = FULL RANK (state = manas_d + manas_prev_g, both bf16, ~2.5 GB at 64 experts; no QR).
    # >0 = the legacy low-rank sketch at that rank. The 137M rank ladder was monotone in rank, so
    # full rank is the limit of the trend at ~zero compute -- and it has never run on a box.
    ap.add_argument("--probe_rank", type=int, default=0)
    # GAMMA TRACKS LR. The probe holds a STANDING displacement of size ~gamma/(1-rho_step); with a
    # fixed gamma that displacement does not shrink as the cosine anneals lr -> 0, so the endgame is
    # optimized at a point the run never lands on. Measured on the MNIST demo: at constant LR the
    # tuned recipe kept its early lead but REGRESSED test acc ~0.02-0.03 at saturation, and setting
    # probe_gamma = law(lr_t) each step recovered the ceiling while keeping the full early lead
    # (the 2D toy showed the same endgame bias first). Our cosine anneals to final_frac 0.0 and bpb
    # is read at the very end, so this is the failure mode this A/B is most exposed to.
    # Safe because gamma is applied at PROBE time and nothing is baked into the buffers -- changing
    # it rescales the dose retroactively (see manas.py _coef_of / _d_of). Set once per STEP, never
    # between micros: apply_probe and _restore_theta must see the same gamma or theta won't restore.
    # WARMUP IS HELD AT THE PEAK DOSE, not ramped: probe warmup was separately measured HARMFUL at
    # BiBo (w100 flipped manas to +0.01-0.02 WORSE than muon -- the win is front-loaded in early
    # descent), so letting gamma follow lr UP from ~0 would rebuild a known-bad arm.
    ap.add_argument("--probe_gamma_schedule", choices=["none", "lr"], default="none")  # tag _gs
    # Log the loss at RESTORED theta every log_every steps (one extra no-grad forward, ~0.3%).
    # Under muon it equals the training loss and is a free consistency check; under manas the
    # difference IS the cosmetic part of the train-loss edge. Deconfound, not an ablation axis --
    # untagged, so it never splits a run name away from its baseline.
    ap.add_argument("--clean_loss", type=_bool, default=False)
    # LatentMoE: shared W_down before dispatch / W_up after combine; experts run at width d.
    # I' / E / k already have flags (--moe_inter / --experts / --top_k), so the matched-budget
    # family is k*I' = const with E = 8k -- see the round config. 0 = off. Run tag _lat<d>.
    ap.add_argument("--wd_schedule", choices=["none", "rcos"], default="none")
    ap.add_argument("--wd_end", type=float, default=0.1)   # only used when --wd_schedule rcos
    ap.add_argument("--scheduler", choices=["wsd", "cosine"], default="wsd")  # LR schedule shape
    ap.add_argument("--warmup_frac", type=float, default=0.05)   # both schedulers
    ap.add_argument("--decay_frac", type=float, default=0.20)    # WSD only: fraction of steps in the final decay
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--dataset", default=TRAIN_DATASET)          # QTK-81K packed instruct corpus (HF id)
    ap.add_argument("--max_steps", type=int, default=1200)   # >0 overrides token budget; 0 = use --tokens
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--hf_repo", default="")     # if set, push model+tokenizer to this HF repo every --ckpt_every steps (async, non-blocking)
    ap.add_argument("--hf_token", default="")    # HF WRITE token; falls back to $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN
    ap.add_argument("--hf_private", action="store_true")  # create the repo private
    # EVAL REMOVED Aug 7 2026 -- these are accepted and IGNORED so existing launch scripts and
    # queue files keep running instead of dying on an unknown flag. Nothing reads them.
    for _dead in ("--eval_every", "--sample_every", "--eval_mcq_n", "--eval_bpb_n",
                  "--final_mcq_n", "--eval_icl_n", "--eval_seq_len"):
        ap.add_argument(_dead, type=int, default=0, help="IGNORED: eval removed")
    for _dead in ("--eval_extrap", "--final_extrap"):
        ap.add_argument(_dead, default="", help="IGNORED: eval removed")
    ap.add_argument("--no_eval_icl", action="store_true", help="IGNORED: eval removed")
    # per-step validation on a FROZEN small batch (2 en + 2 hi instruction-following sources).
    # Logged on the training log line. 0 = follow --log_every, -1 = off.
    ap.add_argument("--val_every", type=int, default=0)
    ap.add_argument("--val_seqs", type=int, default=2)            # sequences per source
    # Skip validation until this step. At init the loss is ~11.4 (uniform over an 81920 vocab), and
    # a single point that high pins every downstream chart's y-axis to it, flattening the range the
    # run actually spends its life in. Purely cosmetic -- the number at step 0 is not informative,
    # it is log(vocab). Set 0 to log from the start.
    ap.add_argument("--val_start", type=int, default=100)
    # END-OF-RUN report on the final weights: 4 fixed prompts (2 en + 2 hi) + degeneration metrics
    # + val CE at each --extrap_lens. Runs after the checkpoint is written, once, so it costs
    # nothing during training and cannot lose the weights if it fails.
    ap.add_argument("--final_report", type=lambda s: str(s).lower() not in ("0", "false", "no"),
                    default=True)
    ap.add_argument("--sample_tokens", type=int, default=96)     # tokens generated per prompt
    # 1024 is the TRAINED length; 2048/4096 are beyond anything RoPE saw. Logged one panel each.
    ap.add_argument("--extrap_lens", default="1024,2048,4096")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="polyglu-ablations")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dt = _DT[args.precision]
    patch_list = [p.strip() for p in args.patches.split(",") if p.strip()]
    use_fused_ce = "ce" in patch_list
    from . import configs as _cfgmod
    _cfgmod.SHARED["norm_topk_prob"] = "sum" if args.norm_topk_prob else False
    if not args.norm_topk_prob:
        print("[router] warning: norm_topk_prob=0 -> NO top-k normalization; raw sigmoid scores are "
              "the combine weights and they will NOT sum to 1.", flush=True)
    if args.pad_id is not None:
        # Warn, do not block: masking id 0 is CORRECT for a corpus delimited by <|im_end|> (81914),
        # and wrong for one delimited by <|endoftext|>. Only the corpus knows which, so state the
        # assumption loudly rather than guess -- the failure is silent (the model never learns to
        # stop) and would read as a mysterious degeneration rather than a bug.
        from transformers import AutoTokenizer
        _eos = AutoTokenizer.from_pretrained(TOKENIZER).eos_token_id
        print(f"[loss] pad_id={args.pad_id} excluded from CE", flush=True)
        if _eos is not None and int(args.pad_id) == int(_eos):
            print(f"[loss] WARNING: pad_id == eos_token_id ({_eos}). Every end-of-text target is "
                  f"now masked. This is safe ONLY if documents are separated by <|im_end|> "
                  f"(81914); if they are separated by <|endoftext|>, the model will never learn "
                  f"to terminate.", flush=True)
    n_total = args.experts or SHARED["num_experts"]
    # --pos_identity_n / --neg_identity_n override special_pairs, so glu_count() alone would print
    # a GLU count the model does not have. A wrong expert-layout line in the log is exactly the
    # kind of thing that gets believed later.
    if args.pos_identity_n is not None or args.neg_identity_n is not None:
        _npos = args.pos_identity_n if args.pos_identity_n is not None else (
            args.special_pairs if args.pos_identity_expert else 0)
        _nneg = args.neg_identity_n if args.neg_identity_n is not None else (
            args.special_pairs if args.neg_identity_expert else 0)
        n_glu = n_total - _npos - _nneg
        _how = f"pos_identity_n={_npos}, neg_identity_n={_nneg}"
    else:
        n_glu = glu_count(n_total, args.special_pairs, args.pos_identity_expert,
                          args.neg_identity_expert)
        _npos = args.special_pairs if args.pos_identity_expert else 0
        _nneg = args.special_pairs if args.neg_identity_expert else 0
        _how = f"special_pairs={args.special_pairs}"
    print(f"[experts] {n_total} routed = {n_glu} GLU ({args.act}) + {_npos} +Identity "
          f"+ {_nneg} -Identity  ({_how})", flush=True)
    assert "moe" in patch_list, "the fused-moe patch is required: eager experts are ~3x slower"
    patchmod.RADIAL_P = args.radial_p
    patchmod.EXPERT_ACT = args.act
    assert args.act == "radial" or "moe" in patch_list, (
        "--act silu steers the PATCHED expert forward; without the 'moe' patch src eager would "
        "silently run radial anyway")
    # src's eager expert path hardcodes sigmoid, so --radial_p tanh is only real on the
    # patched Triton forward. Without the patch the run would silently be a sigmoid run.
    assert args.radial_p == "sigmoid" or "moe" in patch_list, (
        "--radial_p tanh requires the 'moe' patch (src eager is sigmoid-only)")
    _glu_code = (patchmod.SILU_CODE if args.act == "silu"
                 else patchmod.RADIAL_CODES[args.radial_p])
    print(f"[act] expert activation = {args.act}"
          + (f", radial p = {args.radial_p}" if args.act == "radial" else " (plain SwiGLU)")
          + f" -> kernel act code {_glu_code}", flush=True)
    # Shared with run_eval.py so a checkpoint's architecture can be reproduced exactly.
    _swa_pat, _win = resolve_swa(args.swa_pattern, args.sliding_window, SHARED["num_hidden_layers"])
    # None -> SHARED default [0, 9]; "none" -> [] (every layer MoE); else a comma list.
    # "0:32:3:1536" -> {0: {num_routed_experts: 32, num_experts_per_tok: 3, moe_intermediate_size: 1536}}
    _moe_over = {}
    for _spec in filter(None, (x.strip() for x in args.moe_override.split(","))):
        _L, _E, _k, _w = (int(x) for x in _spec.split(":"))
        _moe_over[_L] = {"num_routed_experts": _E, "num_experts_per_tok": _k,
                         "moe_intermediate_size": _w}
        print(f"[ffn] layer {_L} expert geometry override: {_E} experts, top-{_k}, width {_w}",
              flush=True)

    _dense = (None if args.mlp_only_layers is None else
              [] if args.mlp_only_layers.lower() == "none" else
              [int(v) for v in args.mlp_only_layers.split(",")])
    if _dense is not None:
        _bad = [i for i in _dense if not 0 <= i < SHARED["num_hidden_layers"]]
        if _bad:
            raise ValueError(f"--mlp_only_layers {_bad} out of range for "
                             f"{SHARED['num_hidden_layers']} layers")
        print(f"[ffn] dense layers {_dense or 'NONE - every layer is MoE'}", flush=True)
    if _swa_pat is not None:
        print(f"[swa] pattern {_swa_pat} (1=windowed) window={_win} "
              f"qk_norm={args.swa_qk_norm}", flush=True)
    model, cfg = build_arm(args.arm, device=DEV, dtype=torch.float32, attn_impl=args.attn,  # fp32 master
                           bias_update_threshold=args.bias_update_threshold,
                           bias_update_factor=(None if args.bias_update_factor < 0 else args.bias_update_factor),
                           aux_coef=args.aux_coef, num_experts=n_total, special_pairs=args.special_pairs,
                           use_xsa=args.use_xsa,
                           xsa_alpha_init=args.xsa_alpha_init,
                           hybrid_layer_pattern=_swa_pat, sliding_window=_win,
                           swa_qk_norm=args.swa_qk_norm,
                           rope_theta=args.rope_theta,
                           mlp_only_layers=_dense,
                           moe_overrides=_moe_over,
                           intermediate_size=args.dense_inter,
                           attn_res=args.attn_res, attn_res_sites=args.attn_res_sites,
                           attn_res_carry=args.attn_res_carry,
                           use_typed_attn_res=args.typed_attn_res,
                           typed_attn_res_long_memory=args.typed_attn_res_long_memory,
                           typed_attn_res_extra_init=args.typed_attn_res_extra_init,
                           use_typed_attn_res_fast_slow_memory=args.typed_attn_res_fast_slow_memory,
                           typed_attn_res_fast_decay_init=args.typed_attn_res_fast_decay_init,
                           typed_attn_res_slow_decay_init=args.typed_attn_res_slow_decay_init,
                           use_typed_attn_res_innovation_write=args.typed_attn_res_innovation_write,
                           typed_attn_res_innovation_init=args.typed_attn_res_innovation_init,
                           attn_res_fp32_stream=args.attn_res_fp32_stream,
                           attn_res_carry_scale=args.attn_res_carry_scale,
                           attn_res_emb_term=args.attn_res_emb_term,
                           attn_res_emb_scale=args.attn_res_emb_scale,
                           attn_res_emb_site=args.attn_res_emb_site,
                           attn_res_emb_gain=args.attn_res_emb_gain,
                           attn_res_score=args.attn_res_score,
                           attn_res_topk=args.attn_res_topk,
                           num_pos_identity_experts=args.pos_identity_n,
                           num_neg_identity_experts=args.neg_identity_n,
                           attn_res_carry_per_dim=args.attn_res_carry_per_dim,
                           attn_res_carry_gate=args.attn_res_carry_gate,
                           attn_res_emb_per_dim=args.attn_res_emb_per_dim,
                           bf16_residual_stream=args.bf16_residual_stream,
                           bf16_moe_out=args.bf16_moe_out,
                           pos_identity_expert=args.pos_identity_expert,
                           neg_identity_expert=args.neg_identity_expert,
                           top_k=(args.top_k or None), moe_intermediate_size=(args.moe_inter or None),
                           num_shared_experts=args.n_shared)
    aux_collector = _QwenAuxCollector(model) if (args.arm == "qwen" and args.aux_coef > 0) else None
    if args.use_xsa:
        # An --use_xsa arm whose alpha never got built would silently be a control run wearing the
        # xsa tag. Count them rather than trust the config plumbing (that failure has happened).
        _nxa = sum(1 for m in model.modules() if getattr(m, "xsa_alpha", None) is not None)
        assert _nxa, "--use_xsa but 0 modules carry xsa_alpha -- the arm would be silently inert"
        print(f"[xsa] learnable alpha on {_nxa} attn modules, init {args.xsa_alpha_init:g} "
              f"-> tanh = {math.tanh(args.xsa_alpha_init):.3f}", flush=True)
    total, trainable, active = count_params(model)
    patchmod.apply([p for p in patch_list if p != "ce"])              # ce handled in _ce()
    if not args.fused_res_add:
        import exp.modeling_bibo as _E
        _E._HAS_FUSED_RES_ADD = False
        print("[res_add] fused kernel DISABLED by --fused_res_add false; eager carry write",
              flush=True)
    # router mechanics traced on the TRAINING stream (eval-time MoEStats sees eval data only, every
    # eval_every steps). GPU-resident accumulators -> one device->host transfer per log_every.
    # Manas dose law (measured, .autoresearch/manas/): per-vote gamma scales with the vote count and
    # with per-vote gradient noise, and tracks sqrt(LR). k = votes/step = grad_accum, m = micro batch.
    _gamma_law = lambda lr: 0.08 * (lr / 3e-4) ** 0.5 * args.grad_accum / args.batch ** 0.5
    probe_gamma = args.probe_gamma
    if args.optim == "manas" and probe_gamma == 0.0:
        probe_gamma = _gamma_law(args.muon_lr)
        print(f"[optim] manas auto-gamma = 0.08*sqrt({args.muon_lr:g}/3e-4)*{args.grad_accum}"
              f"/sqrt({args.batch}) = {probe_gamma:g}", flush=True)
    if args.optim == "manas" and args.grad_accum < 2:
        print("[optim] WARNING: manas with grad_accum < 2 self-gates to plain Muon (1 vote/step is "
              "the measured info cap) -- slice the batch to get votes", flush=True)
    _n_exp = getattr(cfg, "num_routed_experts", None) or getattr(cfg, "num_experts", 0)
    rtrace = RouterTrace(model, _n_exp, DEV) if (args.router_log and _n_exp >= 2) else None
    if rtrace is not None and getattr(rtrace, "mixed", False):
        print("[router] per-layer expert counts differ (--moe_override): the MODEL-WIDE router "
              "trace is off, since pooling counts across layers with different E is undefined. "
              "The per-layer metrics under interp/router/layer_N/* cover all of it.", flush=True)
    # Per-LAYER router diagnostics: expert-load histogram, balance, boundary gap, logit scale.
    # A model-wide mean cannot say WHICH layer collapsed, and max-load plus entropy are both
    # summary statistics of the very distribution the histogram shows.
    plrouter = (PerLayerRouter(model, _n_exp, args.top_k or SHARED["num_experts_per_tok"])
                if (args.router_log and _n_exp >= 2) else None)
    opts, n_mat, n_oth = build_optimizers(model, args.muon_lr, args.adam_lr, args.wd, ns_dtype=dt,
                                          scale_mode=args.muon_scale_mode, xorth_post=args.xorth_post,
                                          xorth_gate_ref=args.xorth_gate_ref, xorth_ema=args.xorth_ema,
                                          xorth_warmup_steps=args.xorth_warmup_steps, xorth_where=args.xorth_where,
                                          router_adamw=(args.router_optim == "adamw"),
                                          act_scale_lr=args.act_scale_lr,
                                          vec_matrices_adamw=args.vec_matrices_adamw,
                                          vec_adamw_group=args.vec_adamw_group,
                                          cautious_decay=args.cautious_decay,
                                          optim=args.optim, probe_gamma=probe_gamma,
                                          probe_rho_step=args.probe_rho_step,
                                          probe_rank=args.probe_rank)
    if args.compile:                                            # compile the transformer body only; the
        model.model = torch.compile(model.model)               # triton/liger kernels stay eager (compiler.disable)
        print(f"[{args.arm}_seed{args.seed}] torch.compile(model.model) on; fused CE + liger/moe/flash kernels stay eager",
              flush=True)

    tok_per_step = args.batch * args.seq_len * args.grad_accum   # global batch
    total_steps = args.max_steps or (args.tokens // tok_per_step)
    scheds = make_scheduler(args.scheduler, opts, total_steps, args.warmup_frac, args.decay_frac)
    wd_sched = (make_wd_schedule(opts, total_steps, args.wd, args.wd_end)
                if args.wd_schedule == "rcos" else None)
    cur_wd = args.wd
    if wd_sched is not None:
        print(f"[optim] wd schedule rcos: {args.wd:g} -> {args.wd_end:g} (reverse cosine, rises as lr decays)",
              flush=True)
    amp = contextlib.nullcontext() if args.precision == "fp32" else torch.autocast("cuda", dtype=dt)
    # Suffixes exist so variants don't collide on ckpt/log/run names (they otherwise share arm+seed).
    # The acts- tag is gone: radial is the only activation src implements, so every bibo_min run has it.
    run_name = (f"{args.arm}_seed{args.seed}"
                + (("_aS" + f"{args.act_scale_lr:g}") if args.act_scale_lr else "")
                + (("_vecadamw" + ("act" if args.vec_adamw_group == "act" else ""))
                   if args.vec_matrices_adamw else "")
                + ("_ptanh" if args.radial_p == "tanh" else "")
                # XSA MUST be tagged: without it the xsa arm shares a run name with its own
                # control and overwrites its _final.pt / _result.json. That happened once --
                # the control had to be rescued mid-flight by renaming its artifacts.
                + ("_xsa" if args.use_xsa else "")
                + (f"_xaI{args.xsa_alpha_init:g}" if args.use_xsa and args.xsa_alpha_init else "")
                # SWA arms differ from their control by nothing else in the name; untagged they
                # would overwrite the control's _final.pt / _result.json.
                # w128 vs w128-512 must be distinguishable in the filename or the hierarchical
                # arm overwrites the uniform one it is being compared against.
                + (f"_swa{args.swa_pattern}w{str(args.sliding_window).replace(',', '-')}"
                   if args.swa_pattern != "none" else "")
                + ("_noqkn" if args.swa_pattern != "none" and not args.swa_qk_norm else "")
                + (f"_rt{args.rope_theta/1000:g}k" if args.rope_theta else "")
                # A different dense/MoE split is a different model with a different param count;
                # untagged it would overwrite the baseline's ckpt and _result.json on the same seed.
                + (f"_di{args.dense_inter}" if args.dense_inter else "")
                + ("" if not _moe_over else "_mo" + "-".join(
                    f"{L}x{v['num_routed_experts']}k{v['num_experts_per_tok']}w{v['moe_intermediate_size']}"
                    for L, v in sorted(_moe_over.items())))
                + ("" if args.mlp_only_layers is None else
                   "_allmoe" if _dense == [] else "_dense" + "-".join(map(str, _dense)))
                # activation is an axis again; an untagged silu arm would overwrite the radial
                # control's ckpt/_result.json on the same seed+experts
                + (f"_act{args.act}" if args.act != "radial" else "")
                + (f"_ares{args.attn_res}" if args.attn_res != "off" else "")
                + (f"s{args.attn_res_sites}" if args.attn_res != "off"
                   and args.attn_res_sites != 2 else "")
                + ("c" if args.attn_res != "off" and args.attn_res_carry else "")
                + ("_typed" if args.typed_attn_res else "")
                + ("-nolm" if args.typed_attn_res
                   and not args.typed_attn_res_long_memory else "")
                + (f"-x{args.typed_attn_res_extra_init:g}" if args.typed_attn_res
                   and args.typed_attn_res_extra_init != 0.01 else "")
                + (f"-fs{args.typed_attn_res_fast_decay_init:g}"
                   f"-{args.typed_attn_res_slow_decay_init:g}"
                   if args.typed_attn_res_fast_slow_memory else "")
                + (f"-iw{args.typed_attn_res_innovation_init:g}"
                   if args.typed_attn_res_innovation_write else "")
                + ("f32s" if args.attn_res != "off" and args.attn_res_fp32_stream else "")
                + (f"cs{args.attn_res_carry_scale}" if args.attn_res != "off"
                   and args.attn_res_carry_scale != "none" else "")
                + ("" if args.fused_res_add else "_eagerRA")
                + ("emb" if args.attn_res != "off" and args.attn_res_emb_term else "")
                + (args.attn_res_emb_scale if args.attn_res_emb_term
                   and args.attn_res_emb_scale != "none" else "")
                + ("_signorm" if args.attn_res != "off" and args.attn_res_score == "signorm"
                   else "")
                + (f"_top{args.attn_res_topk}" if args.attn_res != "off" and args.attn_res_topk
                   else "")
                + ("_cperdim" if args.attn_res != "off" and args.attn_res_carry_per_dim
                   else "")
                + (f"_cgate{args.attn_res_carry_gate}" if args.attn_res != "off"
                   and args.attn_res_carry_gate != "none" else "")
                + ("_dperdim" if args.attn_res != "off" and args.attn_res_emb_per_dim else "")
                + ("_bf16stream" if args.bf16_residual_stream else "")
                + ("_bf16moeout" if args.bf16_moe_out else "")
                + ("ht" if args.attn_res_emb_term and args.attn_res_emb_site == "ht" else "")
                + ("i" if args.attn_res_emb_term and args.attn_res_emb_site == "ht"
                   and args.attn_res_emb_gain else "")
                # wd is an ablation axis (scale-equilibrium test) -- untagged runs would collide
                # with the wd=0.1 baselines on the same arm+seed and overwrite their ckpt/log names
                + (f"_wdr{args.wd:g}-{args.wd_end:g}" if args.wd_schedule == "rcos"
                   else (f"_wd{args.wd:g}" if args.wd != 0.1 else ""))
                + ("_cwd" if args.cautious_decay else "")
                + (f"_manas-g{probe_gamma:g}" + (f"-r{args.probe_rank}" if args.probe_rank else "")
                   + ("-gs" if args.probe_gamma_schedule == "lr" else "")
                   if args.optim == "manas" else "")
                + (f"_e{n_total}" if n_total != SHARED["num_experts"] else "")
                + (f"_se{args.special_pairs}" if args.special_pairs else "")
                + (f"_id{args.pos_identity_n}p{args.neg_identity_n or 0}n"
                   if args.pos_identity_n is not None else "")
                + (("_posonly" if not args.neg_identity_expert else "") if args.special_pairs else "")
                + (("_negonly" if not args.pos_identity_expert else "") if args.special_pairs else "")
                # bias_update_factor MUST be in the tag: it is a first-class ablation axis (it sets
                # the balancer's authority relative to the router boundary gap), and two arms that
                # differ only in u would otherwise share a run name -- silently overwriting each
                # other's ..._final.pt / _result.json and colliding in W&B. That already happened
                # once: se2-xsp (u=0.001) was clobbered by se2-xsp-u01 (u=0.01).
                + (f"_u{args.bias_update_factor:g}" if args.bias_update_factor >= 0 else "")
                + (f"_k{args.top_k}" if args.top_k else "")
                + (f"_sh{args.n_shared}" if args.n_shared else "")
                + (f"_mi{args.moe_inter}" if args.moe_inter else "")
                + (f"_{args.muon_scale_mode}" if args.muon_scale_mode != "aurora" else "")
                + (f"_xo{args.xorth_post:g}{args.xorth_where}" if args.xorth_post > 0 else "")
                + ("" if args.norm_topk_prob else "_nontp")   # normalization is the default; mark when OFF
                + ("_radamw" if args.router_optim == "adamw" else "")
                + ("_cos" if args.scheduler == "cosine" else "")
                + (f"_{args.run_tag}" if args.run_tag else ""))
    out_dir = args.out or os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(out_dir, exist_ok=True)

    wb = None
    if args.wandb:
        import wandb
        # console="wrap": capture stdout into the run's Logs tab. The end-of-run report prints every
        # greedy AND sampled generation there on purpose -- W&B renders long text poorly in a table,
        # and the log is the readable place for it. Default "auto" can decline to capture when
        # stdout is already redirected, which is exactly how these runs are launched (nohup > log).
        wb = wandb.init(project=args.wandb_project, name=run_name,
                        settings=wandb.Settings(console="wrap"),
                        config={**vars(args), "total_steps": total_steps,
                                "params_total": total, "params_active": active})

    # async HF checkpoint push: save_pretrained locally (main thread), then upload_folder in the background.
    hf_api = hf_tok = None
    hf_futures = []
    if args.hf_repo:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        _hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        hf_api = HfApi(token=_hf_token)
        hf_api.create_repo(args.hf_repo, private=args.hf_private, exist_ok=True)
        hf_tok = AutoTokenizer.from_pretrained(TOKENIZER)   # saved next to the weights so from_pretrained works
        print(f"[{run_name}] HF push -> {args.hf_repo} every {args.ckpt_every} steps (async, non-blocking); "
              f"periodic -> step<N>/, final -> repo root", flush=True)

    # FROZEN validation batches, built once. Failure is FATAL by design: a silently-skipped source
    # would leave the run logging a val number that means something different from every other run.
    val_batches = val_holdout = None
    val_every = args.log_every if args.val_every == 0 else args.val_every
    if args.val_every >= 0 and args.data == "real":
        from transformers import AutoTokenizer
        _vt = AutoTokenizer.from_pretrained(TOKENIZER)
        # TIER 1 -- val/loss, the ranking number: held-out shard of THIS corpus, in-distribution.
        val_holdout = _val.build_holdout(args.dataset, args.seq_len, args.val_seqs, DEV)
        # TIER 2 -- val/ext/*, external instruction sources. Watched, never averaged into val/loss.
        # Separator is <|im_end|> (validation.SEP_ID), NOT eos: the val CE masks the pad id and
        # pad == eos == 0 here, so an eos separator would be deleted from the loss.
        val_batches = _val.build(_vt, args.seq_len, args.val_seqs, DEV)
        if val_holdout is None:
            print("[val] WARNING: no held-out shard (Hub-streamed dataset) -- val/loss is ABSENT "
                  "and only val/ext/* is logged. Nothing in-distribution ranks these arms.", flush=True)
        print(f"[val] every {val_every} steps from step {args.val_start} | val/loss=holdout"
              f"{'' if val_holdout is None else f' {tuple(val_holdout.shape)}'} | "
              f"val/ext: {', '.join(f'{n}/{l}' for n, l, _ in val_batches)}", flush=True)

    print(f"[{run_name}] params total={total/1e6:.2f}M active={active/1e6:.2f}M | steps={total_steps} "
          f"tok/step={tok_per_step} patches={patch_list} {args.precision} attn={args.attn} "
          f"muon_mats={n_mat} eval=REMOVED val={'on' if val_batches else 'off'}", flush=True)

    gen = token_batches(args.batch, args.seq_len, DEV, dataset=args.dataset,
                        synthetic=(args.data == "synthetic"), vocab=cfg.vocab_size, seed=args.seed,
                        pad_id=0 if args.pad_id is None else int(args.pad_id))
    # MFU denominator: measured achievable GEMM peak, or --peak_tflops (theoretical). FLOPs/token = 6N + attn.
    measured_peak = _measure_peak_tflops(DEV, dt)
    peak_tflops = args.peak_tflops if args.peak_tflops > 0 else measured_peak
    flops_per_token = 6 * active + 12 * cfg.num_hidden_layers * cfg.hidden_size * args.seq_len
    print(f"[{run_name}] MFU peak={peak_tflops:.0f} TFLOPS "
          f"({'set' if args.peak_tflops > 0 else 'measured GEMM'}); measured GEMM={measured_peak:.0f} | "
          f"flops/token ~{flops_per_token/1e9:.2f} GFLOP", flush=True)
    model.train()
    t0 = time.time(); _last_t = t0; _last_tok = 0; _last_step = 0
    # Running loss over the last LOSS_WINDOW steps. The per-step `loss` is ONE global batch and
    # swings ~0.3 between adjacent steps, so reading a single step (least of all the last one) is
    # mostly reading batch noise -- every arm comparison should use the window, not the point.
    from collections import deque
    LOSS_WINDOW = 20
    _loss_hist = deque(maxlen=LOSS_WINDOW)
    # Probe hooks, resolved once: under --optim muon these are nullcontext / no-op, so the muon
    # arm's inner loop is byte-identical to every run on the board.
    _mns = opts[0] if hasattr(opts[0], "probe") else None
    _probe = _mns.probe if _mns is not None else contextlib.nullcontext
    _vote = _mns.vote if _mns is not None else (lambda: None)
    # gamma-tracks-lr: hold the peak dose through warmup, then follow the anneal (see the flag).
    _gs_warm = max(int(total_steps * args.warmup_frac), 1)
    _gs_on = _mns is not None and args.probe_gamma_schedule == "lr"
    _clean_on, loss_clean = bool(args.clean_loss), None
    if _gs_on:
        print(f"[optim] manas gamma tracks lr: {probe_gamma:g} held through warmup "
              f"({_gs_warm} steps), then law(lr_t) -> ~0 at the end of the cosine", flush=True)
    for step in range(total_steps):
        if args.typed_attn_res:
            for module in model.modules():
                if getattr(module, "use_typed_attn_res", False):
                    module._typed_diagnostics = (
                        {} if step % args.log_every == 0 or step == total_steps - 1 else None
                    )
        # The per-layer hooks fire only on steps that will actually be logged; every other step
        # pays nothing. One traced step per interval is plenty for a distribution over 64 experts.
        if plrouter is not None:
            plrouter.enabled = (step % args.log_every == 0 or step == total_steps - 1)
        for o in opts:
            o.zero_grad(set_to_none=True)
        if _gs_on and step >= _gs_warm:                      # once per STEP, never between micros
            _mns.probe_gamma = _gamma_law(opts[0].param_groups[0]["lr"])
        loss_val = 0.0
        for _ in range(args.grad_accum):                     # gradient accumulation -> global batch
            ids = next(gen)
            # MANAS: fwd/bwd run at theta + gamma*D (the probe), then vote() folds this micro's
            # gradient direction into D. vote() MUST be outside the probe context (it raises
            # otherwise) and theta is restored exactly inside step(). Both are no-ops under muon.
            with _probe():
                with amp:
                    loss = _ce(model, ids, use_fused_ce, aux_collector, args.aux_coef,
                               getattr(cfg, "num_experts", 6), cfg.num_experts_per_tok,
                               pad_id=args.pad_id) / args.grad_accum
                loss.backward()
            _vote()
            loss_val += loss.item()
        _loss_hist.append(loss_val)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else \
            torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
        # CLEAN-THETA LOSS. Manas logs its training loss at theta + gamma*D, and D is a DESCENT
        # direction, so its train curve is read from downhill of the weights the run actually keeps.
        # The loss landscape is identical between the arms -- what differs is WHERE it is sampled --
        # but the bias scales with the standing probe gamma/(1-rho_step), so it flatters whichever
        # arm probes harder. That is precisely the fixed-vs-gs axis (gamma 0.231 vs 0.098 by the end
        # of a cosine), which is why train loss cannot rank those two. One extra no-grad forward on
        # the last micro-batch at RESTORED theta makes the gap measurable instead of assumed.
        # eval() so RouterTrace skips it (it ignores non-training forwards by design) -- the config
        # has no dropout, so eval mode does not change the computation. ~0.3% overhead at log_every 25.
        if _clean_on and (step % args.log_every == 0 or step == total_steps - 1):
            if _mns is not None:
                _mns._restore_theta()    # step() does this anyway; doing it early is idempotent
            model.eval()
            with torch.no_grad(), amp:
                loss_clean = float(_ce(model, ids, use_fused_ce, None, 0.0,
                                       getattr(cfg, "num_experts", 6), cfg.num_experts_per_tok))
            model.train()
        if wd_sched is not None:
            cur_wd = wd_sched(step)      # BEFORE o.step() so this step decays at the scheduled wd
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        if step % args.log_every == 0 or step == total_steps - 1:
            lv, gn = loss_val, float(gnorm)
            lv_run = sum(_loss_hist) / len(_loss_hist)         # mean over the last LOSS_WINDOW steps
            lr = opts[0].param_groups[0]["lr"]
            toks = (step + 1) * tok_per_step
            _now = time.time(); _dt = _now - _last_t
            _steps_since = max(step - _last_step, 1)
            ms_per_step = 1000.0 * _dt / _steps_since                          # wall time per step
            tps = (toks - _last_tok) / _dt if _dt > 0 else 0.0                 # tokens/sec this interval
            mfu = 100.0 * flops_per_token * tps / (peak_tflops * 1e12) if peak_tflops > 0 else 0.0
            _last_t, _last_tok, _last_step = _now, toks, step
            mem = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
            elapsed = _now - t0                                                # total wall time so far
            eta = (total_steps - step - 1) * elapsed / max(step + 1, 1)        # est. time remaining
            fin = math.isfinite(lv)
            ecorr = _expert_corr(model)                                    # cross-expert redundancy — logged every run
            rcorr = _router_corr(model)                                    # ROUTER expert-direction collapse (the conv-axis metric)
            rt = rtrace.flush() if rtrace else {}                          # router mechanics over the interval
            rt_s = (f" top1w={rt['train/router_top1_weight']:.3f} rent={rt['train/router_entropy']:.3f}"
                    f" bal={rt['train/balance_entropy']:.3f}"
                    f" gap={rt['train/router_boundary_gap']:.4f}"
                    # spl = share of top-k slots on the ±Identity block (neg half in parens); with
                    # --glu_budget r it should settle near 1-r. 0.000 means no special experts.
                    + (f" spl={rt['train/special_load']:.3f}({rt['train/neg_identity_load']:.3f})"
                       if rt.get("train/special_load", 0.0) > 0 else "")) if rt else ""
            # aS = where the radial exponent landed, reported as p = sigmoid(theta) because p is the
            # interpretable quantity: p->0 IS normsilu, p->1 is full magnitude. The trained shape is a
            # DEPTH RAMP (0.11 early -> 0.93 late), so a p pinned near its 0.5 init across every layer
            # means theta is not travelling and --act_scale_lr is too low, NOT that the axis is dead.
            # xa = tanh(xsa_alpha), the APPLIED rejection strength: 0 = XSA off on that head, 1 =
            # full rejection. Reported in tanh units, not logits, because "off" has to be readable
            # at a glance -- an arm that never switches XSA on is a null, not a win.
            rt.update(patchmod.xsa_alpha_stats(model))
            rt.update(_typed_memory_stats(model))
            # Per-tensor param and grad norms. The one that earns its keep is grad/norm_min: a
            # tensor whose gradient is 0.0 while training proceeds is INERT, and this repo has
            # shipped that bug twice (XSA alpha behind a passing parity test, ACT_CYCLE not
            # reaching the eager path). Both were invisible in the loss for days.
            rt.update(tensor_norms(model))
            if plrouter is not None:
                rt.update(plrouter.flush())
                # OFF for the rest of this step. Validation and the extrapolation panels run their
                # own forwards further down, and with the hooks still armed their tokens land in
                # the next interval's expert counts -- 3.9% of the histogram on the board config,
                # measured on base-allmoe-s2026, mixing held-out text into a TRAIN routing metric.
                plrouter.enabled = False
            # XSA alpha and radial p per layer, read straight off the parameters. The depth ramp
            # in p was only ever visible per layer, and the aggregate min/mean/max hid it.
            rt.update(per_layer_params(model))
            xa_s = ((f" xa={rt['train/xsa_a_mean']:+.3f}"
                     f"[{rt['train/xsa_a_min']:+.2f},{rt['train/xsa_a_max']:+.2f}]"
                     if "train/xsa_a_mean" in rt else "")
                    )
            aS_s = xa_s
            typed_s = ""
            if "train/typed_fast_decay_mean" in rt:
                typed_s += (
                    f" memdecay={rt['train/typed_fast_decay_mean']:.3f}"
                    f"/{rt['train/typed_slow_decay_mean']:.3f}"
                )
            if "train/typed_innovation_alpha_mean" in rt:
                typed_s += f" innov={rt['train/typed_innovation_alpha_mean']:.3f}"
            cs_s = ""
            # cs = the learnable carry coefficient, reported as 2*sigmoid(theta) because that is
            # what multiplies attn_output. Init is exactly 1.0, so a cs pinned at 1.000 across
            # every layer means the MLP does not want the knob and it can be removed; a DEPTH
            # PROFILE (early layers wanting mixing, late wanting their own attention) is the
            # result worth having, and is the same shape radial p turned out to have.
            _cs = [m.attn_res_carry_theta for m in model.modules()
                   if getattr(m, "attn_res_carry_theta", None) is not None]
            if _cs:
                _mode = getattr(model.config, "attn_res_carry_scale", "none")
                # THE SAME map the model uses, imported rather than re-spelled. This chain was
                # once a hand-written if/elif ending in an unguarded `else 2*tanh`; when the mode
                # names changed, the new ones fell through to it and s= reported 2*tanh(1.0)=1.523
                # for a c that was exactly 1.0. A second copy of a transform is a second chance to
                # display a number the model never computed.
                from exp.modeling_bibo import _CARRY_C
                assert _mode in _CARRY_C, (
                    f"attn_res_carry_scale={_mode!r} is not in _CARRY_C; s= would report a value "
                    f"the model never used")
                _xf = _CARRY_C[_mode]
                _cl = [_xf(t.detach().float().flatten()) for t in _cs]   # one entry per LAYER
                _c = torch.cat(_cl)
                rt.update({"train/attn_res_s_mean": _c.mean().item(),
                           "train/attn_res_s_min": _c.min().item(),
                           "train/attn_res_s_max": _c.max().item()})
                # PER LAYER too. min/max/mean over 10 layers cannot answer the question this
                # arm exists for -- "do early layers want the depth mix and late layers their
                # own attention?" needs to know WHICH layer, the same way radial p's depth ramp
                # was only visible per layer and its global mean actively misled.
                # With per-dim c each layer holds (hidden,) values, so L{i} carries that layer's
                # MEAN -- enumerating the flat cat would emit 5120 keys named L0..L5119 and
                # destroy the very depth panel this block exists for. The within-layer spread is
                # its own series, because "which layer wants a SHAPED c" is the new question and
                # a mean hides it exactly the way the global mean hid the depth ramp.
                # MEAN, not rms. c is unbounded so channels can go NEGATIVE (the ht-site i already
                # did), and rms discards the sign; it also conflates magnitude with spread, since
                # rms^2 = mean^2 + var, so a layer centred on 0 with a wide spread would read as a
                # large rms. The mean is additionally the quantity that compares directly to the
                # scalar arm, because a scalar IS a uniform vector.
                rt.update({f"train/attn_res_s/L{i}": v.mean().item() for i, v in enumerate(_cl)})
                if _cl[0].numel() > 1:
                    # THE NUMBER THIS ARM TURNS ON. If std stays near 0 the vector has collapsed
                    # back to a scalar, per-dim bought nothing, and any bpb delta is noise -- that
                    # verdict is not visible in the mean. std rather than max-min because one
                    # runaway lane out of 512 would otherwise define the range and hide the
                    # other 511.
                    rt.update({f"train/attn_res_s_std/L{i}": v.std().item()
                               for i, v in enumerate(_cl)})
                    # Per-channel AND unbounded is the first configuration where an individual
                    # channel can flip sign, i.e. the MLP sees that attention channel negated
                    # while the residual stream still gets it unnegated. Cheap to count, and it
                    # is the thing a scalar c could never express.
                    rt.update({f"train/attn_res_s_negfrac/L{i}": (v < 0).float().mean().item()
                               for i, v in enumerate(_cl)})
                    _sd = torch.stack([v.std() for v in _cl])
                    rt["train/attn_res_s_std_mean"] = _sd.mean().item()
                    rt["train/attn_res_s_std_max"] = _sd.max().item()
                    _sd_tag = f" sd={_sd.mean().item():.3f}"
                else:
                    _sd_tag = ""
                # AFTER the assignment below, never before it: cs_s is reset here every log step
                # (it starts as "" up at the top), so appending first put the tag on a string that
                # was then thrown away. W&B still got the keys -- they go into rt -- so this was
                # only ever a missing console tag, not lost data.
                cs_s = (f" s={_c.mean().item():.3f}"
                        f"[{_c.min().item():.2f},{_c.max().item():.2f}]" + _sd_tag)
                aS_s = xa_s + cs_s        # covers the no-radial case; the radial block below
                                          # rebuilds from xa_s + cs_s so neither clobbers the other
            # d = the embedding-skip knob, PER LAYER. The whole arm is a depth-profile question --
            # "does layer 1 alone want a token-identity channel?" -- so a mean would answer nothing.
            # Note d does not exist at layer 0, so L index here is layer_idx-1.
            # attn_res_emb_gain (the unit-norm i) logs under the SAME attn_res_d/* keys on purpose:
            # it is the same axis re-parameterized, and sharing the keys keeps one wandb panel
            # across arms. They are mutually exclusive by construction -- the gain REPLACES theta --
            # so there is never a collision. Read the console tag (d= vs i=) for which one it is;
            # the units differ (theta is a logit, i is directly the injected rms).
            # GATE. There is no bias, so there is no static component to report -- the whole
            # coefficient is SiLU(W @ attn_read) and W is the only parameter. Its RMS is the
            # "is input-dependence actually being used" signal, the analogue of s_std for the
            # static per-dim arm: standard init puts it near 1/sqrt(H) = 0.044, so growth means
            # the gate is learning to discriminate and a flat trace means it is not.
            # The gate OUTPUT distribution is what we really want, but reading it needs a hook in
            # the forward and a sync every step; it is recoverable from the checkpoint afterwards.
            _gm = [m for m in model.modules() if getattr(m, "_gate_mean", None) is not None]
            if _gm:
                # SAME PANEL as the scalar and per-dim arms. The gate output is c, so it goes to
                # attn_res_s/* and attn_res_s_std/*, not to keys of its own -- the depth profile
                # is only readable if every parameterization of c plots on one axis. No collision
                # is possible: the gate requires carry_scale=none, so the theta-based block above
                # found nothing and left these keys free.
                _mu = torch.stack([m._gate_mean for m in _gm])
                _sd = torch.stack([m._gate_std for m in _gm])
                rt.update({f"train/attn_res_s/L{i}": v for i, v in enumerate(_mu.tolist())})
                rt.update({f"train/attn_res_s_std/L{i}": v for i, v in enumerate(_sd.tolist())})
                rt.update({"train/attn_res_s_mean": _mu.mean().item(),
                           "train/attn_res_s_min": min(m._gate_min.item() for m in _gm),
                           "train/attn_res_s_max": max(m._gate_max.item() for m in _gm),
                           "train/attn_res_s_std_mean": _sd.mean().item(),
                           "train/attn_res_s_std_max": _sd.max().item()})
                # W's RMS is a DIFFERENT quantity -- a parameter norm, not a coefficient value --
                # so it keeps its own key rather than polluting the s axis it is 25x below.
                # It answers "is the input-dependence being used at all": standard init is
                # ~1/sqrt(H)=0.044, and a flat trace means the gate is behaving as a constant.
                _r = torch.stack([m.attn_res_carry_gate.detach().float().pow(2).mean().sqrt()
                                  for m in _gm])
                rt.update({f"train/attn_res_gate_w/L{i}": v for i, v in enumerate(_r.tolist())})
                rt["train/attn_res_gate_w_mean"] = _r.mean().item()
                cs_s += (f" s={_mu.mean().item():.3f}"
                         f"[{min(m._gate_min.item() for m in _gm):.2f},"
                         f"{max(m._gate_max.item() for m in _gm):.2f}]"
                         f" sd={_sd.mean().item():.3f} gw={_r.mean().item():.4f}")
                aS_s = xa_s + cs_s
            _de = [m.attn_res_emb_theta for m in model.modules()
                   if getattr(m, "attn_res_emb_theta", None) is not None]
            _dtag = "d"
            if not _de:
                _de = [m.attn_res_emb_gain for m in model.modules()
                       if getattr(m, "attn_res_emb_gain", None) is not None]
                _dtag = "i"
            if _de:
                _d = torch.cat([t.detach().float().flatten() for t in _de])
                rt.update({"train/attn_res_d_mean": _d.mean().item(),
                           "train/attn_res_d_min": _d.min().item(),
                           "train/attn_res_d_max": _d.max().item()})
                # L{i+1} carries that layer's MEAN (d does not exist at layer 0). With per-dim d
                # each layer holds (hidden,) values, so enumerating the flat cat would emit keys
                # L1..L5120 and destroy the depth panel -- the same trap per-dim c hit.
                rt.update({f"train/attn_res_d/L{i + 1}": t.detach().float().mean().item()
                           for i, t in enumerate(_de)})
                _dsd = torch.stack([t.detach().float().flatten().std() for t in _de]) \
                    if _de[0].numel() > 1 else None
                if _dsd is not None:
                    rt.update({f"train/attn_res_d_std/L{i + 1}": v
                               for i, v in enumerate(_dsd.tolist())})
                    rt["train/attn_res_d_std_mean"] = _dsd.mean().item()
                    rt["train/attn_res_d_std_max"] = _dsd.max().item()
                    # Symmetric with per-dim c. Both per-dim c arms measured negfrac = 0 at every
                    # layer -- no attention channel ever wanted to be negated. d is a different
                    # question: it weights the EMBEDDING, and a channel wanting the token identity
                    # subtracted is not obviously absurd the way a negated attention channel was.
                    rt.update({f"train/attn_res_d_negfrac/L{i + 1}":
                               (t.detach() < 0).float().mean().item()
                               for i, t in enumerate(_de)})
                cs_s += (f" {_dtag}={_d.mean().item():.3f}"
                         f"[{_d.min().item():.2f},{_d.max().item():.2f}]"
                         + (f" dsd={_dsd.mean().item():.3f}" if _dsd is not None else ""))
                aS_s = xa_s + cs_s
            _th = [m.radial_theta for m in model.modules() if hasattr(m, "radial_theta")]
            if _th:
                _t = torch.cat([t.detach().float().flatten() for t in _th])
                _p = torch.sigmoid(_t)
                # BOTH units, on purpose. train/act_alpha_* is raw THETA under the key the pre-Aug-1
                # runs used -- keeping it is what lets a new run overlay the old ones on the same
                # W&B panel (bibo-act-1b). train/radial_p_* is sigmoid(theta), the interpretable
                # one: p->0 IS normsilu, p->1 full magnitude. Logging only p silently breaks the
                # comparison twice over -- new key AND new scale, so 0.500 vs 0.000 at step 0 reads
                # as a changed init when the tensor is identical (both zeros).
                rt.update({"train/act_alpha_mean": _t.mean().item(),
                           "train/act_alpha_min": _t.min().item(),
                           "train/act_alpha_max": _t.max().item(),
                           "train/radial_p_mean": _p.mean().item(),
                           "train/radial_p_min": _p.min().item(),
                           "train/radial_p_max": _p.max().item()})
                aS_s = (xa_s + cs_s + f" p={rt['train/radial_p_mean']:.3f}"
                        f"[{rt['train/radial_p_min']:.2f},{rt['train/radial_p_max']:.2f}]"
                        f" th={rt['train/act_alpha_mean']:+.3f}")
            # VALIDATION on the frozen batch. Unlike train loss this is the same text every step and
            # every arm, so it is the only number here that can rank two runs.
            val_s, val_flat = "", {}
            if ((val_holdout is not None or val_batches) and step >= args.val_start
                    and (step % val_every == 0 or step == total_steps - 1)):
                _vo, val_flat = _val.losses(model, val_holdout, val_batches,
                                            fused_linear_cross_entropy, amp,
                                            pad_id=0 if args.pad_id is None else int(args.pad_id))
                # headline first and on its own; the external sources follow, tagged, so the two
                # tiers can never be misread as one aggregate.
                val_s = ("" if _vo is None else f" val={_vo:.4f}") + (
                    " ext[" + " ".join(f"{k.rsplit('/', 1)[1][:2]}={v:.2f}"
                                       for k, v in val_flat.items()
                                       if k.startswith("val/ext/") and k.rsplit("/", 1)[1]
                                       not in ("en", "hi")) + "]" if val_batches else "")
            print(f"  step {step}/{total_steps} loss={lv:.4f} run{len(_loss_hist)}={lv_run:.4f} |g|={gn:.3f} lr={lr:.2e} tok={toks/1e6:.1f}M "
                  f"ms/step={ms_per_step:.0f} tps={tps/1e3:.1f}k mfu={mfu:.1f}% mem={mem:.1f}G "
                  f"xcorr={ecorr:.4f} rcorr={rcorr:.4f}{val_s}{rt_s}{aS_s}{typed_s}"
                  f"{f' wd={cur_wd:.4f}' if wd_sched is not None else ''}"
                  # clean = loss at RESTORED theta; probe = how much of `loss` is the probe sitting
                  # downhill. Under muon probe is ~0 by construction (nothing to restore).
                  + (f" clean={loss_clean:.4f} probe={loss_clean - lv:+.4f}"
                     if loss_clean is not None else "")
                  + f" elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                  f"{'' if fin else '  <<NON-FINITE>>'}", flush=True)
            if wb:
                wb.log({"train/loss": lv, "train/grad_norm": gn, "train/lr": lr, "train/ms_per_step": ms_per_step,
                        "train/tps": tps, "train/mfu": mfu, "train/mem_gb": mem, "train/elapsed_s": elapsed,
                        "train/expert_corr": ecorr, "train/router_corr": rcorr,
                        **({"train/loss_clean": loss_clean, "train/probe_gap": loss_clean - lv}
                           if loss_clean is not None else {}),
                        **({"train/probe_gamma": _mns.probe_gamma} if _mns is not None else {}),
                        "tokens": toks, **_interp(rt), **val_flat}, step=step)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            if hf_api is not None:
                _dir = _save_hf_ckpt(model, hf_tok, os.path.join(out_dir, f"{run_name}_step{step}"))
                hf_futures.append(_push_hf_async(hf_api, args.hf_repo, _dir, f"step{step}",
                                                 f"{run_name} step{step}"))
            else:
                torch.save(model.state_dict(), os.path.join(out_dir, f"{run_name}_step{step}.pt"))

    ckpt = os.path.join(out_dir, f"{run_name}_final.pt")
    torch.save(model.state_dict(), ckpt)

    # ---- END-OF-RUN REPORT on the FINAL weights: samples + degeneration + length extrapolation.
    # After the checkpoint is written, never before: a bug in reporting must not cost the run its
    # weights. final_report swallows its own exceptions for the same reason.
    final_flat = {}
    if args.final_report and args.data == "real":
        from . import final_report as _fr
        from transformers import AutoTokenizer as _AT
        print(f"\n[{run_name}] final report: {len(_fr.PROMPTS)} prompts + extrapolation "
              f"at {args.extrap_lens}", flush=True)
        try:
            _rt = _AT.from_pretrained(TOKENIZER)
            _lens = tuple(int(x) for x in str(args.extrap_lens).split(",") if x.strip())
            final_flat = _fr.run(model, _rt, args.dataset, fused_linear_cross_entropy, amp,
                                 device=DEV, wb=wb, max_new=args.sample_tokens,
                                 n_seqs=args.val_seqs, lens=_lens)
        except Exception as _e:
            print(f"[{run_name}] final report FAILED: {type(_e).__name__}: {_e}", flush=True)
    if hf_api is not None:                                  # final -> repo root so `from_pretrained(repo)` just works
        _dir = _save_hf_ckpt(model, hf_tok, os.path.join(out_dir, f"{run_name}_final"))
        hf_futures.append(_push_hf_async(hf_api, args.hf_repo, _dir, "final", f"{run_name} final"))
    res = {"arm": args.arm, "seed": args.seed, "steps": total_steps, "tokens": total_steps * tok_per_step,
           "final_loss": loss_val, "final_loss_running": sum(_loss_hist) / len(_loss_hist), "params_total": total, "params_active": active,
           "ckpt": ckpt, "wall_s": time.time() - t0, "eval": None,
           # the W&B run id, so report_ckpt.py can RESUME this run rather than opening a second one.
           # A report logged to its own run cannot be compared against the training curves it
           # describes, which is the entire point of logging it.
           "wandb_id": (getattr(wb, "id", None) if wb else None),
           "wandb_project": (args.wandb_project if wb else None),
           "final_report": final_flat,
           "config": vars(args)}
    with open(os.path.join(out_dir, f"{run_name}_result.json"), "w") as f:
        json.dump(res, f, indent=2)
    if hf_futures:                                         # drain: block until every background upload lands,
        print(f"[{run_name}] waiting on {len(hf_futures)} HF upload(s) before exit...", flush=True)
        for f in hf_futures:                               # else this detached process exits and kills the threads
            try:
                f.result()
            except Exception:
                pass
    if wb:
        wb.finish()
    final_loss_run = sum(_loss_hist) / len(_loss_hist)
    print(f"[{run_name}] DONE final_loss={loss_val:.4f} final_loss_run{len(_loss_hist)}={final_loss_run:.4f} "
          f"ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()

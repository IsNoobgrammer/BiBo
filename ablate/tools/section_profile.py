"""Where a training step's GPU time goes, by MODEL SECTION and direction (forward / backward).

    python -m ablate.common.train <board flags> --section_profile 30

Sections: attention split into global and sliding-window layers, the layer-0 all-active ensemble
(E == top_k) vs the routed MoE layers (each INCLUDING its pre-norm, which the megakernel fuses in),
the fused CE, the optimizer step, and "other" (embedding, AttnRes mixing/carry, grad clip, glue).

CUDA events only, no host syncs, so the timed step runs exactly as an untimed one. The seams are
wrapped per INSTANCE: self_attn.forward, and the layer's FFN method (_attn_res_mlp_forward /
_standard_ffn_forward / _ffn_forward) -- NOT mlp.forward, because the megakernel patch never calls
the MoE module. Backward bounds are tensor hooks: grad of the seam's OUTPUT arriving = its backward
starts, grad of its INPUT arriving = its backward is done. Summed over all micro-batches.
"""
import collections

import torch

_FFN_SEAMS = ("_attn_res_mlp_forward", "_standard_ffn_forward", "_ffn_forward")


class SectionProfiler:
    def __init__(self, model):
        self.ev = collections.defaultdict(list)     # label -> [(start_event, end_event), ...]
        self.marks = {}
        self._undo = []
        body = getattr(model, "model", model)
        self._h = [body.register_forward_hook(self._trunk_post)]
        for layer in body.layers:
            at = layer.self_attn
            self._wrap(at, "forward", "attention sliding-window" if getattr(at, "is_swa", False) else "attention global")
            mlp = layer.mlp
            e, k = getattr(mlp, "num_routed_experts", None), getattr(mlp, "num_experts_per_tok", None)
            lab = ("dense MLP" if not hasattr(mlp, "experts") else
                   "L0 ensemble (all-active)" if e is not None and e == k else "MoE routed")
            for seam in _FFN_SEAMS:
                if hasattr(layer, seam):
                    self._wrap(layer, seam, lab)

    @staticmethod
    def _event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _trunk_post(self, m, args, out):                      # CE fwd = trunk end .. ce_end mark
        self.mark("trunk_end")
        h = getattr(out, "last_hidden_state", None)
        h = out[0] if h is None else h
        if torch.is_tensor(h) and h.requires_grad:              # CE bwd = ce_end .. this grad arriving
            h.register_hook(lambda g: self.mark("trunk_bwd_start") or None)

    def _wrap(self, obj, name, label):
        fn = getattr(obj, name)
        st = {}

        def timed(*args, **kwargs):
            x = next((a for a in args if torch.is_tensor(a)), kwargs.get("hidden_states"))
            if torch.is_tensor(x) and x.requires_grad:          # input grad arriving = backward done
                x.register_hook(lambda g: self.ev[label + " | bwd"].append((st.pop("b0"), self._event()))
                                if "b0" in st else None)
            f0 = self._event()
            out = fn(*args, **kwargs)
            self.ev[label + " | fwd"].append((f0, self._event()))
            y = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(y) and y.requires_grad:          # output grad arriving = backward starts
                y.register_hook(lambda g: st.__setitem__("b0", self._event()) or None)
            return out

        setattr(obj, name, timed)
        self._undo.append((obj, name))

    def mark(self, name):
        self.marks.setdefault(name, []).append(self._event())

    def remove(self):
        for h in self._h:
            h.remove()
        for obj, name in self._undo:
            delattr(obj, name)                                  # back to the class method

    def report(self, pairs):
        """pairs: {label: (start_mark, end_mark)} over the marks recorded with mark()."""
        torch.cuda.synchronize()
        ms = collections.OrderedDict()
        for lab, lst in sorted(self.ev.items()):
            ms[lab] = sum(a.elapsed_time(b) for a, b in lst)
        for lab, (m0, m1) in pairs.items():
            ms[lab] = sum(a.elapsed_time(b) for a, b in zip(self.marks.get(m0, []), self.marks.get(m1, [])))
        total = self.marks["step_start"][0].elapsed_time(self.marks["step_end"][0])
        ms["other (emb, AttnRes, clip, glue)"] = total - sum(ms.values())
        print(f"\n[section] one training step, {total:.1f} ms GPU wall (all micro-batches)")
        print(f"[section] {'section':46s} {'ms':>9s} {'share':>7s}")
        for lab, v in sorted(ms.items(), key=lambda kv: -kv[1]):
            print(f"[section] {lab:46s} {v:9.1f} {100 * v / total:6.1f}%")
        fb = collections.defaultdict(float)
        for lab, v in ms.items():
            fb[lab.split(" | ")[0]] += v
        print("[section] --- fwd + bwd combined")
        for lab, v in sorted(fb.items(), key=lambda kv: -kv[1]):
            print(f"[section] {lab:46s} {v:9.1f} {100 * v / total:6.1f}%")
        return ms

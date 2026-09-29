"""Where a training step's GPU time goes, by MODEL SECTION and direction (forward / backward).

    python -m ablate.common.train <board flags> --section_profile 30

Sections: attention split into global and sliding-window layers, the layer-0 all-active ensemble
(E == top_k) vs the routed MoE layers, the fused CE, the optimizer step, and "other" (embedding,
norms outside the modules, AttnRes mixing/carry, grad clip -- everything not inside a hooked module).

CUDA events only, no host syncs, so the timed step runs exactly as an untimed one. Forward bounds
are module pre/post hooks. Backward bounds are tensor hooks: the grad of a module's OUTPUT arriving
= its backward starts, the grad of its INPUT arriving = its backward is done. Summed over all
micro-batches of the step.
"""
import collections

import torch


class SectionProfiler:
    def __init__(self, model):
        self.ev = collections.defaultdict(list)     # label -> [(start_event, end_event), ...]
        self.marks = {}
        self._h = []
        body = getattr(model, "model", model)

        def trunk_post(m, args, out):                           # CE fwd = trunk end .. ce_end mark
            self.mark("trunk_end")
            h = getattr(out, "last_hidden_state", None)
            h = out[0] if h is None else h
            if torch.is_tensor(h) and h.requires_grad:          # CE bwd = ce_end .. this grad arriving
                h.register_hook(lambda g: self.mark("trunk_bwd_start") or None)

        self._h.append(body.register_forward_hook(trunk_post))
        for i, layer in enumerate(body.layers):
            at = layer.self_attn
            self._hook(at, "attention global" if not getattr(at, "is_swa", False) else "attention sliding-window")
            mlp = layer.mlp
            e = getattr(mlp, "num_routed_experts", None)
            k = getattr(mlp, "num_experts_per_tok", None)
            dense = not hasattr(mlp, "experts")
            lab = ("dense MLP" if dense else
                   "L0 ensemble (all-active)" if (e is not None and k is not None and e == k) else "MoE routed")
            self._hook(mlp, lab)

    @staticmethod
    def _event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _hook(self, mod, label):
        st = {}

        def pre(m, args):
            st["f0"] = self._event()
            x = next((a for a in args if torch.is_tensor(a) and a.requires_grad), None)
            if x is not None:                                   # input grad arriving = backward done
                x.register_hook(lambda g: self.ev[label + " | bwd"].append((st.pop("b0"), self._event())) or None)

        def post(m, args, out):
            self.ev[label + " | fwd"].append((st.pop("f0"), self._event()))
            y = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(y) and y.requires_grad:          # output grad arriving = backward starts
                y.register_hook(lambda g: st.__setitem__("b0", self._event()) or None)

        self._h += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]

    def mark(self, name):
        """Named point in the step; pairs are turned into sections in report()."""
        self.marks.setdefault(name, []).append(self._event())

    def remove(self):
        for h in self._h:
            h.remove()

    def report(self, pairs):
        """pairs: {label: (start_mark, end_mark)} over the marks recorded with mark()."""
        torch.cuda.synchronize()
        ms = collections.OrderedDict()
        for lab, lst in sorted(self.ev.items()):
            ms[lab] = sum(a.elapsed_time(b) for a, b in lst)
        for lab, (m0, m1) in pairs.items():
            ms[lab] = sum(a.elapsed_time(b) for a, b in zip(self.marks.get(m0, []), self.marks.get(m1, [])))
        total = self.marks["step_start"][0].elapsed_time(self.marks["step_end"][0])
        ms["other (emb, AttnRes, norms, clip, glue)"] = total - sum(ms.values())
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

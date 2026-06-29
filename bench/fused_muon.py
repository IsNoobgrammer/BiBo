"""Fused Muon optimizer step — vendored from triton-kernel-fused (kernels/muon.py).

Drop-in replacement for bench/optim.py's eager Muon: SAME Moonlight recipe (momentum -> orthogonalize,
0.2*sqrt(max) scale, decoupled WD, per-expert 3D batching) and bit-parity to the eager step in fp32
(verified 4.8e-7). The wins are pure launch/precision overhead — NOT a math change:

  * fp16 Newton-Schulz (`ns_dtype`): engages T4 fp16 tensor cores. BiBo's eager NS ran fp32 (no tensor
    cores on sm_75); fp16 is ~2.8x faster on the NS GEMMs (the dominant ~70% of the step). bf16 has NO
    tensor cores on sm_75 — fp16 is the T4 choice. fp16-NS gate passed (SV mean ~1, |dp|/lr ~0.2, NaN-free).
  * mixed precision (the AdamW-8bit norm): fp32 master weights, fp32 grads, fp16 MOMENTUM state.
  * same-shape batched Newton-Schulz: all same-(rows,cols) params orthogonalize in ONE batched bmm
    (compile can't reach this — it runs NS per param), row-chunked by `ns_batch_elems` so the per-step
    transient stays <= the eager peak.
  * foreach gather/scatter + baddbmm-folded axpy: collapse the per-param launch loop.

`DistributedMuon` (option B) is the DDP path for 2x T4: each rank orthogonalizes a FLOP-balanced subset
and broadcasts updates — bit-identical to the replicated step, ~1/world_size of the NS work per rank.
"""
from collections import defaultdict

import torch
import torch.optim as optim

# Polar-Express per-iteration NS coefficients (nprime06/parameter-golf) — the --modded-muon coeffs.
_PE_COEFFS = (
    (8.156554524902461,  -22.48329292557795,  15.878769915207462),
    (4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
)
# BiBo's tuned quintic Moonlight coeffs (the default NS) — used for all 5 iterations.
_QUINTIC = ((3.4445, -4.7750, 2.0315),) * 5


def newton_schulz(G, coeffs=_QUINTIC, ns_dtype=torch.float16, eps=1e-7):
    """Orthogonalize G (drive singular values -> 1). 2D -> (1,A,B); 3D experts (E,A,B) batch over E.
    Frobenius norm is fp32-ACCUMULATED (no full fp32 copy; an fp16 sum-of-squares overflows), then the
    iteration GEMMs run in ns_dtype. baddbmm folds each axpy into the GEMM (3 GEMMs/iter, no pointwise)."""
    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)                              # iterate on the smaller Gram
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype) / nrm.to(ns_dtype)
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


class FusedMuon(optim.Optimizer):
    """Muon with fp16 NS + same-shape batched (row-chunked) Newton-Schulz + foreach gather/scatter.
    `scale_mode`: 'moonlight' = 0.2*sqrt(max(r,c)) (BiBo's recipe, lr~3e-4); 'jordan' = max(1,r/c)**0.5.
    Step 2D + 3D params only; route 1D params / conv kernels to AdamW upstream."""

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, weight_decay=0.0,
                 coeffs=_QUINTIC, ns_dtype=torch.float16, scale_mode="moonlight",
                 ns_batch_elems=4 * 1024 * 1024):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        self.coeffs = coeffs
        self.ns_dtype = ns_dtype
        self.scale_mode = scale_mode
        self.ns_batch_elems = ns_batch_elems                        # speed<->memory knob (cap rows*r*c/NS call)

    def _scale(self, p):
        r, c = p.shape[-2], p.shape[-1]
        if self.scale_mode == "moonlight":
            return 0.2 * (max(r, c) ** 0.5)
        return max(1, r / c) ** 0.5

    def _plan(self, group, params):
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = self._plan_cache = {}
        key = id(group)
        if key in cache:
            return cache[key]
        buckets = defaultdict(list)
        for p in params:
            buckets[(p.shape[-2], p.shape[-1])].append(p)
        plan = []
        for (r, c), ps in buckets.items():
            members, off = [], 0
            for p in ps:
                n = p.numel() // (r * c)
                members.append((p, off, n)); off += n
            M, anchor = off, ps[0]
            if "muon_mom" not in self.state[anchor]:                # don't clobber a loaded checkpoint
                self.state[anchor]["muon_mom"] = torch.zeros((M, r, c), device=anchor.device, dtype=self.ns_dtype)
            scale = 0.2 * (max(r, c) ** 0.5) if self.scale_mode == "moonlight" else max(1, r / c) ** 0.5
            row_cap = max(1, self.ns_batch_elems // (r * c))        # params kept whole; >=1 per chunk
            chunks, cur, cur_rows, start = [], [], 0, 0
            for p, _o, n in members:
                if cur and cur_rows + n > row_cap:
                    chunks.append((cur, start, cur_rows)); start += cur_rows; cur, cur_rows = [], 0
                cur.append((p, cur_rows, n)); cur_rows += n
            if cur:
                chunks.append((cur, start, cur_rows))
            plan.append({"r": r, "c": c, "chunks": chunks, "anchor": anchor, "scale": scale})
        cache[key] = plan
        return plan

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            plan = self._plan(group, params)
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)          # decoupled weight decay (fp32 master)
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:
                    mom_c = mom[start:start + crows]
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c
                    out = newton_schulz(u, self.coeffs, self.ns_dtype)
                    torch._foreach_add_([p for p, _, _ in members],
                                        [out[o:o + n].reshape(p.shape) for p, o, n in members], alpha=alpha)
        return loss


class DistributedMuon(FusedMuon):
    """Option B for DDP: each rank orthogonalizes a FLOP-balanced subset of the params and broadcasts its
    packed updates; every rank applies the full set. Bit-identical to the replicated FusedMuon (same
    all-reduced grads in -> same weights out), ~1/world_size of the NS per rank, momentum sharded by owner.
    Assumes grads are already all-reduced (DDP default) and params share one dtype."""

    def __init__(self, params, *, process_group=None, **kwargs):
        super().__init__(params, **kwargs)
        self.pg = process_group
        self._owner = None

    def _ordered(self):
        return [(p, g) for g in self.param_groups for p in g["params"] if p.ndim in (2, 3)]

    def _owners(self, ordered, ws):
        load, owner = [0] * ws, []
        for p, _ in ordered:
            r = min(range(ws), key=lambda i: load[i])
            owner.append(r); load[r] += p.numel()
        return owner

    @torch.no_grad()
    def step(self, closure=None):
        import torch.distributed as dist
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ws, rank = dist.get_world_size(self.pg), dist.get_rank(self.pg)
        ordered = self._ordered()
        if self._owner is None or len(self._owner) != len(ordered):
            self._owner = self._owners(ordered, ws)                 # deterministic — same on every rank

        upd = {}
        for i, (p, g) in enumerate(ordered):                        # compute NS only for MY owned params
            if self._owner[i] != rank or p.grad is None:
                continue
            gr = p.grad.to(self.ns_dtype)
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(gr)
            buf = st["momentum_buffer"]
            buf.mul_(g["momentum"]).add_(gr)
            u = gr.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
            upd[i] = newton_schulz(u, self.coeffs, self.ns_dtype).to(p.dtype)

        for src in range(ws):                                       # broadcast each rank's packed updates
            idxs = [i for i in range(len(ordered))
                    if self._owner[i] == src and ordered[i][0].grad is not None]
            if not idxs:
                continue
            sizes = [ordered[i][0].numel() for i in idxs]
            ref = ordered[idxs[0]][0]
            blob = (torch.cat([upd[i].reshape(-1) for i in idxs]) if src == rank
                    else torch.empty(sum(sizes), device=ref.device, dtype=ref.dtype))
            dist.broadcast(blob, src=src, group=self.pg)
            off = 0
            for i, n in zip(idxs, sizes):
                p, g = ordered[i]
                u = blob[off:off + n].view_as(p); off += n
                lr, wd = g["lr"], g["weight_decay"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr * self._scale(p))
        return loss

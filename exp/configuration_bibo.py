"""Experimental BiBo configuration with Kimi K3 Block AttnRes.

The stable configuration remains in ``src.configuration_bibo``. This module is
intentionally a one-way extension: ``exp`` may reuse ``src``, while ``src`` has
no dependency on the experimental package.
"""

from src.configuration_bibo import BiBoConfig as _StableBiBoConfig

__all__ = ["BiBoConfig"]


class BiBoConfig(_StableBiBoConfig):
    """BiBo configuration for the experimental Attention Residuals model.

    ``attn_res_block_size`` is measured in complete transformer decoder layers,
    matching the official Kimi K3 Hugging Face implementation. K3 uses 12.
    Passing ``None`` disables AttnRes and provides a standard-residual control.
    """

    model_type = "bibo_attn_res"

    def __init__(self, attn_res_block_size=12, attn_res_sites=2, attn_res_carry=False,
                 attn_res_fp32_stream=False, attn_res_carry_scale="none",
                 attn_res_emb_term=False, attn_res_emb_scale="none",
                 attn_res_emb_site="mlp", attn_res_emb_gain=False,
                 attn_res_score="softmax", attn_res_topk=0, attn_res_carry_per_dim=False,
                 attn_res_carry_gate="none", attn_res_emb_per_dim=False, **kwargs):
        if attn_res_sites not in (1, 2):
            raise ValueError(
                f"attn_res_sites must be 2 (K3 faithful: a depth-mix before the attention "
                f"sublayer AND before the MLP sublayer) or 1 (one mix per layer at the layer "
                f"input; the MLP takes an ordinary PreNorm residual), got {attn_res_sites!r}")
        self.attn_res_sites = attn_res_sites
        self.attn_res_carry = bool(attn_res_carry)
        self.attn_res_fp32_stream = bool(attn_res_fp32_stream)
        _csm = "none" if attn_res_carry_scale in (False, None) else str(attn_res_carry_scale)
        if _csm not in ("none", "raw", "unbounded", "rms"):
            raise ValueError(f"attn_res_carry_scale must be one of none/raw/rms, "
                             f"got {attn_res_carry_scale!r}")
        self.attn_res_carry_scale = _csm
        # A THIRD, OFF-SIMPLEX term on the carry write: h = attn_read + c*attn_out + d*embedding,
        # d learnable per layer, init 0 (so it is a strict generalization of plain carry).
        # The embedding is ALREADY a candidate inside attn_read, but that read is a convex
        # combination -- weighting the embedding costs prefix_sum weight one for one, so the model
        # cannot ask for MORE total token identity, only for a different split. d is additive and
        # unconstrained, which is the same reason c exists.
        # Motivation: layer 1 runs XSA BACKWARDS (tanh(alpha) to -0.95, i.e. it nearly doubles the
        # self-value component) in every s1 arm measured, while setting its carry c to ~0.03-0.09,
        # the smallest in the model. Reading: it wants token identity in the STREAM and has no
        # channel for it, so it abuses XSA. If that is right, d spikes at layer 1 and alpha there
        # returns positive. The depth profile of d IS the experiment -- do not hard-code it to L1.
        self.attn_res_emb_term = bool(attn_res_emb_term)
        self.attn_res_emb_scale = str(attn_res_emb_scale)
        self.attn_res_emb_site = str(attn_res_emb_site)
        self.attn_res_emb_gain = bool(attn_res_emb_gain)
        # HOW the depth scores become weights. Both options are CONVEX COMBINATIONS -- the depth
        # weights sum to 1 either way, so the read stays a weighted average and the residual
        # stream cannot grow with the candidate count. That constraint is deliberate.
        #   softmax  x_i -> exp(x_i)/sum exp(x_j).  SHIFT-INVARIANT: only score differences
        #            matter, so N candidates carry N-1 usable degrees of freedom.
        #   signorm  x_i -> sigmoid(x_i)/sum sigmoid(x_j). Shift-SENSITIVE, so all N are live,
        #            and it saturates toward uniform once every score is large and positive.
        # Note what this does NOT do: it does not free the model from the simplex, so it is not
        # a substitute for c/d/i, which exist precisely because weighting one candidate costs
        # another its weight one for one. Those stay.
        if attn_res_score not in ("softmax", "signorm"):
            raise ValueError(f"attn_res_score must be softmax or signorm, got {attn_res_score!r}")
        self.attn_res_score = str(attn_res_score)
        # SPARSE DEPTH. 0 = dense. k > 0 mixes only the k best-scoring candidates, prefix_sum
        # forced in. k is effectively a DEPTH THRESHOLD, not a width: it is inert wherever the
        # candidate pool is <= k, and the pool grows with depth. k=1 would select the prefix sum
        # alone at every site, deleting the depth mix entirely, so it is rejected rather than
        # silently building a model with no AttnRes in it.
        attn_res_topk = int(attn_res_topk or 0)
        if attn_res_topk < 0 or attn_res_topk == 1:
            raise ValueError(f"attn_res_topk must be 0 (dense) or >= 2, got {attn_res_topk!r}")
        self.attn_res_topk = attn_res_topk
        # c as an (hidden,) vector instead of one scalar per layer. Strict generalization:
        # every entry inits to the scalar's init, so step 0 is bit-identical. Only meaningful
        # with a learnable carry scale -- at attn_res_carry_scale=none there is no parameter.
        if attn_res_carry_per_dim and not attn_res_carry:
            raise ValueError("attn_res_carry_per_dim needs attn_res_carry=True")
        if attn_res_carry_per_dim and _csm == "none":
            raise ValueError(
                "attn_res_carry_per_dim needs a learnable attn_res_carry_scale; at 'none' the "
                "carry is a fixed ones buffer and the flag would be silently inert.")
        self.attn_res_carry_per_dim = bool(attn_res_carry_per_dim)
        # c = SiLU(W @ attn_read): per token AND per channel, where per_dim is per channel only.
        # No bias -- the router's balancing bias is the only one in this architecture and it
        # stays that way. SiLU rather than sigmoid because the static c it replaces reached
        # 2.133, which a sigmoid gate cannot represent at all.
        # Refused with per_dim or a learnable carry_scale: the gate already produces the whole
        # coefficient, so either would put two knobs on one number.
        _g = "none" if attn_res_carry_gate in (False, None) else str(attn_res_carry_gate)
        if _g == "True":
            _g = "full"
        if _g not in ("none", "diag", "full"):
            raise ValueError(f"attn_res_carry_gate must be none/diag/full, got {_g!r}")
        if _g != "none":
            if not attn_res_carry:
                raise ValueError("attn_res_carry_gate needs attn_res_carry=True")
            if attn_res_carry_per_dim:
                raise ValueError("attn_res_carry_gate subsumes attn_res_carry_per_dim; pick one")
            if _csm != "none":
                raise ValueError(
                    f"attn_res_carry_gate produces the whole coefficient, so "
                    f"attn_res_carry_scale must be none, got {_csm!r}")
        self.attn_res_carry_gate = _g
        # d as an (hidden,) vector, the same widening per-dim c got. Needs the emb term to
        # exist, and is meaningless on the ht gain path where theta is not created at all.
        if attn_res_emb_per_dim:
            if not attn_res_emb_term:
                raise ValueError("attn_res_emb_per_dim needs attn_res_emb_term=True")
            if attn_res_emb_gain:
                raise ValueError(
                    "attn_res_emb_per_dim is for the theta-based emb term; the ht gain path "
                    "replaces theta with i and would ignore it")
        self.attn_res_emb_per_dim = bool(attn_res_emb_per_dim)
        # bf16_residual_stream is deliberately NOT a named parameter here. It is owned by the
        # PARENT (src BiBoConfig) and reaches it through **kwargs. Naming it here silently broke
        # the flag: this __init__ set the attribute, then super().__init__(**kwargs) ran the
        # parent's __init__, which no longer saw the key (this signature had swallowed it) and
        # reset the attribute to its own default False. cfg.bf16_residual_stream came back False
        # for every arm that asked for True, with no error anywhere.
        if attn_res_block_size is not None and (
            isinstance(attn_res_block_size, bool)
            or not isinstance(attn_res_block_size, int)
            or attn_res_block_size <= 0
        ):
            raise ValueError(
                "attn_res_block_size must be a positive integer number of decoder layers "
                f"or None, got {attn_res_block_size!r}"
            )
        self.attn_res_block_size = attn_res_block_size
        super().__init__(**kwargs)

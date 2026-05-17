import { SectionHeader } from '@/components/SectionHeader';
import { PlotImage } from '@/components/PlotImage';
import { Tidbit } from '@/components/Tidbit';

export function SpecialExperts() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="✦"
        title="BiBo Special Expert Analysis"
        description="BiBo's architectural innovation: not all experts are MLPs. Identity, Zero, and ReLU² experts provide diverse computational primitives that the router can select based on token needs."
      />

      {/* Main plot */}
      <div className="glass rounded-xl p-5">
        <PlotImage src="special_expert_analysis" alt="Special Expert Analysis" />
      </div>

      {/* Expert Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="glass rounded-xl p-5 border-t-2 border-t-emerald-500/50">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🪞</span>
            <h3 className="text-sm font-bold text-white">Identity Expert</h3>
          </div>
          <div className="space-y-2 text-xs text-white/60">
            <p><span className="font-mono text-emerald-400">output = input</span></p>
            <p className="text-white/40">Usage: ~13.4% of tokens (E5)</p>
          </div>
          <Tidbit variant="insight" title="Purpose">
            Learned skip connection. When a token already has a good representation,
            the router can choose &quot;do nothing&quot; for this slot — preserving the residual stream
            without adding noise from an unnecessary MLP transformation. The other top-K
            experts still contribute their weighted outputs alongside Identity.
          </Tidbit>
        </div>

        <div className="glass rounded-xl p-5 border-t-2 border-t-violet-500/50">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">⊘</span>
            <h3 className="text-sm font-bold text-white">Zero Expert</h3>
          </div>
          <div className="space-y-2 text-xs text-white/60">
            <p><span className="font-mono text-violet-400">output = 0</span></p>
            <p className="text-white/40">Usage: ~13.2% of tokens (E6)</p>
          </div>
          <Tidbit variant="insight" title="Purpose">
            When Zero is selected as one of the top-3, that slot contributes nothing —
            but the <strong>other 2 selected experts + the shared Conv1D</strong> still contribute
            their full weighted outputs. Zero effectively says &quot;I don&apos;t need a 3rd expert
            for this token — 2 experts + shared conv is enough.&quot; Saves compute for one slot.
          </Tidbit>
        </div>

        <div className="glass rounded-xl p-5 border-t-2 border-t-amber-500/50">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">⚡</span>
            <h3 className="text-sm font-bold text-white">ReLU² Expert</h3>
          </div>
          <div className="space-y-2 text-xs text-white/60">
            <p><span className="font-mono text-amber-400">output = ReLU(x)²</span></p>
            <p className="text-white/40">Usage: ~12.0% of tokens (E7)</p>
          </div>
          <Tidbit variant="insight" title="Purpose">
            Sparse activation. ReLU² produces sharper, more selective features than SwiGLU.
            Useful for tokens that need strong feature selection — the squaring amplifies
            large activations and suppresses small ones more aggressively.
          </Tidbit>
        </div>
      </div>

      {/* Why this matters */}
      <div className="glass rounded-xl p-5 border border-emerald-500/10">
        <h3 className="text-sm font-bold text-white mb-3">Why Heterogeneous Experts Matter</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-xs font-semibold text-bibo-400 uppercase tracking-wider mb-2">BiBo approach</h4>
            <p className="text-xs text-white/60 leading-relaxed">
              The router chooses the <strong>right tool for each slot</strong>:
              full MLP for complex tokens needing transformation,
              Identity for already-good representations,
              Zero when 2 experts + shared conv is sufficient,
              ReLU² for tokens needing sharp feature selection.
              Remember: top-3 routing means each token gets 3 experts + shared conv.
              Zero/Identity in one slot doesn&apos;t mean the token gets nothing — it means
              the other 2 slots + conv handle it.
            </p>
          </div>
          <div>
            <h4 className="text-xs font-semibold text-qwen-400 uppercase tracking-wider mb-2">Qwen approach</h4>
            <p className="text-xs text-white/60 leading-relaxed">
              ALL tokens go through identical SwiGLU MLPs regardless of complexity.
              A padding token gets the same expensive computation as a critical
              semantic token. No way to say &quot;this token is simple, skip the heavy processing
              in one slot.&quot; Every slot always pays full MLP cost.
            </p>
          </div>
        </div>
        <Tidbit variant="insight" title="Compute efficiency">
          Zero expert = 0 FLOPs. Identity = 0 FLOPs. ReLU² = ~1/3 FLOPs of SwiGLU MLP.
          With ~38% of token-expert slots going to cheap experts (13.4% Identity + 13.2% Zero + 12.0% ReLU²),
          BiBo saves significant MoE compute per token while the other selected experts and shared conv
          still provide full processing capacity.
        </Tidbit>
      </div>
    </div>
  );
}

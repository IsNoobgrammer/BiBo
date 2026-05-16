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
            <p className="text-white/40">Usage: ~16% of tokens</p>
          </div>
          <Tidbit variant="insight" title="Purpose">
            Learned skip connection. When a token already has a good representation,
            the router can choose &quot;do nothing&quot; — preserving the residual stream
            without adding noise from an unnecessary MLP transformation.
          </Tidbit>
        </div>

        <div className="glass rounded-xl p-5 border-t-2 border-t-violet-500/50">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">⊘</span>
            <h3 className="text-sm font-bold text-white">Zero Expert</h3>
          </div>
          <div className="space-y-2 text-xs text-white/60">
            <p><span className="font-mono text-violet-400">output = 0</span></p>
            <p className="text-white/40">Usage: ~19% of tokens (highest)</p>
          </div>
          <Tidbit variant="insight" title="Purpose">
            Dump bucket. When Zero is selected, only the always-active shared Conv1D
            contributes. This means &quot;for this token, local context (conv) is sufficient —
            no global MLP needed.&quot; Saves compute for &quot;easy&quot; tokens.
          </Tidbit>
        </div>

        <div className="glass rounded-xl p-5 border-t-2 border-t-amber-500/50">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">⚡</span>
            <h3 className="text-sm font-bold text-white">ReLU² Expert</h3>
          </div>
          <div className="space-y-2 text-xs text-white/60">
            <p><span className="font-mono text-amber-400">output = ReLU(x)²</span></p>
            <p className="text-white/40">Usage: ~14% of tokens</p>
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
              The router chooses the <strong>right tool for each token</strong>:
              full MLP for complex tokens needing transformation,
              Identity for already-good representations,
              Zero for tokens where only local context matters,
              ReLU² for tokens needing sharp feature selection.
              This is why BiBo achieves lower loss with fewer parameters.
            </p>
          </div>
          <div>
            <h4 className="text-xs font-semibold text-qwen-400 uppercase tracking-wider mb-2">Qwen approach</h4>
            <p className="text-xs text-white/60 leading-relaxed">
              ALL tokens go through identical SwiGLU MLPs regardless of complexity.
              A padding token gets the same expensive computation as a critical
              semantic token. No way to say &quot;this token is simple, skip the heavy processing.&quot;
              This wastes parameters and compute on tokens that don&apos;t need it.
            </p>
          </div>
        </div>
        <Tidbit variant="insight" title="Compute efficiency">
          Zero expert = 0 FLOPs. Identity = 0 FLOPs. ReLU² = ~1/3 FLOPs of SwiGLU MLP.
          When ~49% of tokens route to cheap experts (16% Identity + 19% Zero + 14% ReLU²),
          BiBo saves ~35% of MoE compute while achieving better results. Free lunch.
        </Tidbit>
      </div>
    </div>
  );
}

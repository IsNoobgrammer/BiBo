import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { PlotImage } from '@/components/PlotImage';
import { Tidbit } from '@/components/Tidbit';

export function Balance() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="⊞"
        title="Load Balance Analysis"
        description="How evenly tokens are distributed across experts. Lower Gini = more balanced. Higher entropy = more uniform. Balance ratio = min_load / max_load (1.0 = perfect)."
      />

      {/* Summary Plot */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Load Balance Summary — All Configurations</h3>
        <PlotImage src="load_balance_summary" alt="Load Balance Summary" />
        <Tidbit variant="insight" title="Key takeaway">
          BiBo&apos;s balance <em>improves</em> with sequence length — more tokens give the bias heuristics
          more signal to converge. At seq128+, BiBo achieves near-perfect balance without any auxiliary loss.
          Qwen&apos;s balance stays roughly constant regardless of sequence length.
        </Tidbit>
      </div>

      {/* Side by Side Usage Sweeps */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-bibo-400 mb-3">BiBo — Usage Sweep</h3>
          <PlotImage src="usage_sweep_BiBo" alt="BiBo Usage Sweep" />
          <Tidbit variant="bibo" title="What you see">
            Expert usage across batch sizes (1→64) and seq lengths. At seq128/bs64, balance ratio
            reaches 0.88 with Gini=0.019. The bias heuristics converge with sufficient tokens.
          </Tidbit>
          <Tidbit variant="bibo" title="Why it matters">
            Balance is <strong>sequence-length dependent</strong> — longer sequences give the bias heuristics
            more opportunities to correct imbalances. This is ideal for long-context applications.
          </Tidbit>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-qwen-400 mb-3">Qwen3MoE — Usage Sweep</h3>
          <PlotImage src="usage_sweep_Qwen3MoE" alt="Qwen Usage Sweep" />
          <Tidbit variant="qwen" title="What you see">
            Surprisingly uneven: Gini=0.076 at seq128/bs64, balance_ratio=0.68.
            Some experts consistently get 15% while others get only 10%.
          </Tidbit>
          <Tidbit variant="qwen" title="Why it matters">
            The aux loss creates a gradient signal toward uniformity, but the softmax router
            develops <strong>persistent favorites</strong> that the loss can&apos;t fully correct.
            The loss fights the router — and partially loses.
          </Tidbit>
        </div>
      </div>

      {/* Comparative Usage */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Comparative Usage by Sequence Length</h3>
        <SeqTabs prefix="comparative_usage" suffix="bs64" />
        <Tidbit variant="neutral" title="How to read">
          Each bar shows per-expert token allocation. BiBo&apos;s heterogeneous experts (Identity, Zero, ReLU²)
          naturally attract different fractions — but the overall distribution is more uniform than Qwen&apos;s
          supposedly &quot;balanced&quot; homogeneous experts.
        </Tidbit>
      </div>

      {/* Weight Rank + Diversity */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white/80 mb-3">Weight Rank Distribution</h3>
          <SeqTabs prefix="weight_rank_distribution" />
          <Tidbit variant="neutral" title="Interpretation">
            Shows average weight given to rank-1, rank-2, rank-3 expert. Flatter = all top-K experts
            contribute equally. BiBo&apos;s router normalization ensures the rank-2 and rank-3 experts
            get meaningful weight, not just crumbs from rank-1.
          </Tidbit>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white/80 mb-3">Routing Diversity</h3>
          <SeqTabs prefix="routing_diversity" suffix="bs64" />
          <Tidbit variant="neutral" title="Interpretation">
            Effective expert count = exp(H). Higher = more experts meaningfully used per token.
            With top-3 routing, the theoretical max is 3.0. BiBo gets closer to this ceiling.
          </Tidbit>
        </div>
      </div>
    </div>
  );
}

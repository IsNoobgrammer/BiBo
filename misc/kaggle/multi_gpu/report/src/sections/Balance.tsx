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
        <Tidbit variant="bibo" title="Primary driver: bias heuristics">
          BiBo&apos;s near-perfect balance is <strong>primarily due to the bias heuristic</strong> —
          the non-trainable router bias that gets updated every 800 tokens (bias_update_threshold).
          When an expert is underused, its bias increases; when overused, it decreases. This simple
          mechanism achieves Gini≈0.02 without any auxiliary loss gradient interference.
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen: aux loss can&apos;t fix persistent favorites">
          Despite the auxiliary load-balancing loss, Qwen maintains Gini≈0.24 and CV≈0.44 across
          ALL configurations. The aux loss creates a gradient toward uniformity, but the softmax
          router develops <strong>persistent favorites</strong> (E0, E2, E7 get 20%+ while E1 gets 5%)
          that the loss can&apos;t fully correct. The loss fights the router — and loses.
        </Tidbit>
      </div>

      {/* Side by Side Usage Sweeps */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-bibo-400 mb-3">BiBo — Usage Sweep</h3>
          <PlotImage src="usage_sweep_BiBo" alt="BiBo Usage Sweep" />
          <Tidbit variant="bibo" title="What you see">
            Expert usage across batch sizes (1→64) and seq lengths. At seq256/bs64, balance ratio
            reaches 0.90 with Gini=0.022. The bias heuristics converge with sufficient tokens.
          </Tidbit>
          <Tidbit variant="bibo" title="Why bias heuristics work better than aux loss">
            The bias heuristic is a <strong>direct correction</strong> — it literally adds/subtracts
            from router logits to steer tokens toward underused experts. No gradient approximation,
            no interference with the task loss. It&apos;s a control loop, not an optimization objective.
          </Tidbit>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-qwen-400 mb-3">Qwen3MoE — Usage Sweep</h3>
          <PlotImage src="usage_sweep_Qwen3MoE" alt="Qwen Usage Sweep" />
          <Tidbit variant="qwen" title="What you see">
            Extremely uneven: Gini=0.235 at seq256/bs64, balance_ratio=0.24.
            Some experts get 21% while others get only 5% — a 4× imbalance.
          </Tidbit>
          <Tidbit variant="qwen" title="Why aux loss fails here">
            The aux loss is an <strong>indirect signal</strong> — it adds a penalty to the total loss,
            hoping the optimizer will adjust router weights to balance load. But the task loss gradient
            pushes toward using the &quot;best&quot; experts, creating a tug-of-war. With only 8 experts and
            top-3 routing, the task loss wins and creates permanent favorites.
          </Tidbit>
        </div>
      </div>

      {/* Comparative Usage */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Comparative Usage by Sequence Length</h3>
        <SeqTabs prefix="comparative_usage" suffix="bs64" />
        <Tidbit variant="neutral" title="How to read">
          Each bar shows per-expert token allocation. BiBo&apos;s heterogeneous experts (Identity, Zero, ReLU²)
          naturally attract different fractions — but the overall distribution is far more uniform than Qwen&apos;s
          supposedly &quot;balanced&quot; homogeneous experts. The red dashed line shows the ideal uniform allocation.
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

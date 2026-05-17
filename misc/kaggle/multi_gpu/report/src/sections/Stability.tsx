import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { Tidbit } from '@/components/Tidbit';
import { MetricCard } from '@/components/MetricCard';

export function Stability() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="◎"
        title="Routing Stability"
        description="How consistent are routing decisions across different inputs? Measured via Jaccard similarity of expert sets across 20 random samples. Higher = same token always routes the same way."
      />

      {/* Stability scores */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-white/40 mb-4">Average Stability Scores</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="BiBo seq64" value="0.391" variant="bibo" />
          <MetricCard label="BiBo seq256" value="0.392" variant="bibo" />
          <MetricCard label="Qwen seq64" value="0.929" variant="qwen" />
          <MetricCard label="Qwen seq256" value="0.934" variant="qwen" />
        </div>
        <Tidbit variant="neutral" title="What stability means">
          Stability measures whether the same token embedding routes to the same experts
          regardless of context. A score of 1.0 = perfectly deterministic (same token → same experts always).
          A score of 0.0 = completely random. Neither extreme is ideal.
        </Tidbit>
      </div>

      {/* Side by side */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-bibo-400 mb-3">BiBo — Stability (avg: 0.39)</h3>
          <SeqTabs prefix="routing_stability_BiBo" />
          <Tidbit variant="bibo" title="Lower stability = more adaptive">
            The conv router makes <strong>context-dependent</strong> decisions — the same token
            gets different experts depending on its neighbors. This is by design: a token
            meaning &quot;3&quot; should route differently when surrounded by [1,2,3] vs [7,8,3].
          </Tidbit>
          <Tidbit variant="bibo" title="Correlated with task awareness">
            BiBo&apos;s low stability directly enables the sorted-vs-unsorted routing shift
            visible in the position-type analysis. The same token value routes to Identity
            (green) in unsorted positions but ReLU² (orange) in sorted positions — impossible
            with deterministic routing.
          </Tidbit>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-qwen-400 mb-3">Qwen3MoE — Stability (avg: 0.93)</h3>
          <SeqTabs prefix="routing_stability_Qwen3MoE" />
          <Tidbit variant="qwen" title="Very high stability = rigid routing">
            The linear router sees only the current token embedding → same embedding almost always
            routes the same way regardless of context. Stability of 0.93 means routing is nearly
            deterministic — the router has essentially memorized a fixed token→expert mapping.
          </Tidbit>
          <Tidbit variant="qwen" title="Consequence: no task phase awareness">
            Qwen&apos;s sorted-vs-unsorted routing is nearly identical — the router can&apos;t distinguish
            &quot;understanding input&quot; from &quot;generating output.&quot; Same token → same experts regardless
            of whether the model is reading or writing. This rigidity limits expressiveness.
          </Tidbit>
        </div>
      </div>

      {/* Insight */}
      <div className="glass rounded-xl p-5 border border-emerald-500/10">
        <Tidbit variant="insight" title="The stability-performance connection">
          BiBo&apos;s lower stability (0.39) isn&apos;t just a number — it&apos;s the mechanism behind three
          observable behaviors: (1) the dramatic sorted-vs-unsorted routing shift where special
          experts change roles by task phase, (2) the more uniform weight allocation across all
          3 selected experts (visible in the KDE plots), and (3) the better load balance since
          context-dependent routing prevents any single expert from becoming a &quot;permanent favorite.&quot;
          Qwen&apos;s 0.93 stability creates the opposite: rigid favorites, top-1 dominance in weights,
          and no task-phase awareness.
        </Tidbit>
      </div>
    </div>
  );
}

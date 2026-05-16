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
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          <MetricCard label="BiBo seq64" value="0.31" variant="bibo" />
          <MetricCard label="BiBo seq128" value="0.30" variant="bibo" />
          <MetricCard label="BiBo seq256" value="0.32" variant="bibo" />
          <MetricCard label="Qwen seq64" value="0.44" variant="qwen" />
          <MetricCard label="Qwen seq128" value="0.47" variant="qwen" />
          <MetricCard label="Qwen seq256" value="0.49" variant="qwen" />
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
          <h3 className="text-sm font-bold text-bibo-400 mb-3">BiBo — Stability (avg: 0.31)</h3>
          <SeqTabs prefix="routing_stability_BiBo" />
          <Tidbit variant="bibo" title="Lower stability = more adaptive">
            The conv router makes <strong>context-dependent</strong> decisions — the same token
            gets different experts depending on its neighbors. This is by design: a token
            meaning &quot;3&quot; should route differently when surrounded by [1,2,3] vs [7,8,3].
          </Tidbit>
          <Tidbit variant="bibo" title="Trade-off">
            Lower stability makes routing harder to predict and debug. But it enables
            richer representations — the model can process the same token differently
            based on local context, similar to how attention is context-dependent.
          </Tidbit>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-qwen-400 mb-3">Qwen3MoE — Stability (avg: 0.47)</h3>
          <SeqTabs prefix="routing_stability_Qwen3MoE" />
          <Tidbit variant="qwen" title="Higher stability = more predictable">
            The linear router sees only the current token embedding → same embedding always
            routes the same way regardless of context. This makes routing deterministic
            and easy to analyze.
          </Tidbit>
          <Tidbit variant="qwen" title="Trade-off">
            Higher stability means the router can&apos;t adapt to context. Token &quot;3&quot; always
            goes to the same expert whether it&apos;s in [1,2,3] or [7,8,3]. This limits
            the model&apos;s expressiveness in the MoE layers.
          </Tidbit>
        </div>
      </div>

      {/* Insight */}
      <div className="glass rounded-xl p-5 border border-emerald-500/10">
        <Tidbit variant="insight" title="The stability-performance trade-off">
          BiBo&apos;s lower stability (0.31) correlates with its lower loss (0.10).
          Context-dependent routing is more expressive — it allows the model to use
          different expert combinations for the same token based on surrounding context.
          This is analogous to how attention weights change based on context.
          The cost is reduced interpretability and harder debugging.
        </Tidbit>
      </div>
    </div>
  );
}

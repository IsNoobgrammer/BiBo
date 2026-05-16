import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { Tidbit } from '@/components/Tidbit';

export function Heatmaps() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="▦"
        title="Expert Selection Heatmaps"
        description="Which expert is selected (top-1) at each token position, per layer. Color = expert ID. Patterns reveal whether routing is structured or random."
      />

      {/* Side by side heatmaps */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-bibo-400 mb-3">BiBo — Expert Heatmap</h3>
          <SeqTabs prefix="expert_heatmap_v2_BiBo" />
          <Tidbit variant="bibo" title="Spatial patterns">
            Notice the <strong>horizontal bands</strong> — the conv router creates local coherence.
            Consecutive tokens tend to route to the same expert, forming &quot;processing chunks.&quot;
            This is analogous to how CNNs create receptive fields — the router groups tokens
            that should be processed together.
          </Tidbit>
          <Tidbit variant="bibo" title="Layer variation">
            Different layers show different patterns. Early layers have broader bands (coarser grouping),
            later layers have finer patterns (more token-specific routing). The router learns
            hierarchical grouping.
          </Tidbit>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-qwen-400 mb-3">Qwen3MoE — Expert Heatmap</h3>
          <SeqTabs prefix="expert_heatmap_v2_Qwen3MoE" />
          <Tidbit variant="qwen" title="Random-looking patterns">
            More <strong>salt-and-pepper</strong> — the linear router has no local context.
            Each token is routed independently based solely on its embedding, with no awareness
            of neighboring tokens. Adjacent tokens can route to completely different experts.
          </Tidbit>
          <Tidbit variant="qwen" title="Implication">
            Without local coherence, Qwen can&apos;t learn &quot;this group of tokens should be processed
            together.&quot; Every token is an island. This limits the model&apos;s ability to learn
            phrase-level or chunk-level patterns in the MoE layers.
          </Tidbit>
        </div>
      </div>

      {/* Position-Type Routing */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-4">Position-Type Routing: Unsorted vs Sorted Tokens</h3>
        <p className="text-xs text-white/50 mb-4">
          The sorting task has two phases: unsorted input tokens and sorted output tokens.
          Does the router treat them differently?
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-xs font-bold text-bibo-400 mb-2">BiBo</h4>
            <SeqTabs prefix="position_type_routing_BiBo" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-qwen-400 mb-2">Qwen3MoE</h4>
            <SeqTabs prefix="position_type_routing_Qwen3MoE" />
          </div>
        </div>
        <Tidbit variant="insight" title="Position awareness">
          BiBo&apos;s conv router can distinguish input from output positions because it sees
          local context — the transition from unsorted to sorted creates a detectable pattern.
          This allows different expert strategies for &quot;understanding&quot; vs &quot;generating.&quot;
        </Tidbit>
      </div>
    </div>
  );
}

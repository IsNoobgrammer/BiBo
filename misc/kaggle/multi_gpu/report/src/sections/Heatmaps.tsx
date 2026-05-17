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

      {/* Position-Type Routing — the key insight */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-4">Position-Type Routing: Unsorted vs Sorted Tokens</h3>
        <p className="text-xs text-white/50 mb-4">
          The sorting task has two phases: unsorted input tokens and sorted output tokens.
          Does the router treat them differently? Color coding for BiBo: <span className="text-emerald-400 font-semibold">green = Identity</span>, <span className="text-gray-400 font-semibold">grey = Zero</span>, <span className="text-orange-400 font-semibold">orange/red = ReLU²</span>.
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
        <Tidbit variant="bibo" title="BiBo: dramatic sorted-vs-unsorted shift">
          The special experts completely change roles between task phases. On <strong>unsorted input</strong>,
          Identity (green) spikes — the router says &quot;these tokens are already fine, just pass them through
          while I read.&quot; On <strong>sorted output</strong>, ReLU² (orange) spikes — the router says
          &quot;I need sharp feature selection to generate the correct sorted value.&quot; The router has learned
          the task structure without being told about it.
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen: nearly identical routing for both phases">
          Qwen&apos;s routing is <strong>almost the same</strong> for unsorted input and sorted output.
          The same experts dominate in both phases (E0, E2, E3, E7 in L1–L3). The router can&apos;t
          distinguish &quot;understanding&quot; from &quot;generating&quot; because it only sees individual token
          embeddings — no context about where in the sequence it is or what phase the task is in.
        </Tidbit>
        <Tidbit variant="insight" title="This is the conv router&apos;s killer feature">
          BiBo&apos;s conv router (kernel=3) sees the transition from unsorted to sorted tokens as a
          detectable local pattern change. This enables different expert strategies per phase —
          cheap Identity for reading, expensive ReLU²/MLP for generating. Qwen&apos;s linear router
          is blind to this transition. This phase-awareness directly explains BiBo&apos;s better
          generalization to unseen sequence lengths.
        </Tidbit>
      </div>
    </div>
  );
}

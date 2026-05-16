import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { Tidbit } from '@/components/Tidbit';

export function Switching() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="⇄"
        title="Expert Switching Rate"
        description="How often does the top-1 expert change between consecutive tokens? High switching = position-sensitive routing with no local coherence. Low switching = 'sticky' routing where the same expert handles token chunks."
      />

      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Switching Rate by Sequence Length</h3>
        <SeqTabs prefix="switching_rate" />

        <div className="mt-4 grid md:grid-cols-3 gap-3">
          <div className="bg-white/[0.02] rounded-lg p-3 border border-white/5">
            <p className="text-xs text-white/40 uppercase tracking-wider mb-1">Random baseline</p>
            <p className="text-sm text-white/70 font-mono">7/8 = 0.875</p>
            <p className="text-[10px] text-white/40 mt-1">If routing were random, 87.5% of consecutive tokens would switch experts</p>
          </div>
          <div className="bg-bibo-500/5 rounded-lg p-3 border border-bibo-500/10">
            <p className="text-xs text-bibo-400/70 uppercase tracking-wider mb-1">BiBo switching</p>
            <p className="text-sm text-bibo-400 font-mono">&lt; 0.875</p>
            <p className="text-[10px] text-white/40 mt-1">Below random → &quot;sticky&quot; routing. Conv router creates local coherence.</p>
          </div>
          <div className="bg-qwen-500/5 rounded-lg p-3 border border-qwen-500/10">
            <p className="text-xs text-qwen-400/70 uppercase tracking-wider mb-1">Qwen switching</p>
            <p className="text-sm text-qwen-400 font-mono">≈ 0.875</p>
            <p className="text-[10px] text-white/40 mt-1">Near random → no local coherence. Each token routed independently.</p>
          </div>
        </div>

        <Tidbit variant="bibo" title="Why sticky routing helps">
          When consecutive tokens route to the same expert, that expert can build up
          &quot;momentum&quot; — its internal state (via the shared Conv1D) captures local patterns.
          This is similar to how RNNs process sequences: continuity enables pattern detection.
        </Tidbit>
        <Tidbit variant="qwen" title="Why random switching is limiting">
          Without local coherence, each expert sees isolated tokens with no sequential context.
          The model must rely entirely on attention for sequential understanding — the MoE layer
          can&apos;t contribute to sequence-level patterns.
        </Tidbit>
        <Tidbit variant="insight" title="Connection to conv router">
          BiBo&apos;s lower switching rate is a direct consequence of the Conv1D router (kernel=3).
          The router sees a window of 3 tokens, so if tokens 5-6-7 have similar hidden states,
          they&apos;ll likely route to the same expert. This creates natural &quot;processing chunks.&quot;
        </Tidbit>
      </div>
    </div>
  );
}

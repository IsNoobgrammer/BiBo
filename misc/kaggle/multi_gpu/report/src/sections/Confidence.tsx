import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { Tidbit } from '@/components/Tidbit';

export function Confidence() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="↗"
        title="Routing Confidence"
        description="How confident is the router in its top-1 choice? High confidence means top-1 dominates and the other K-1 experts are wasted. Moderate confidence means all selected experts contribute meaningfully."
      />

      {/* Confidence Evolution */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Confidence Evolution Over Token Position</h3>
        <SeqTabs prefix="confidence_evolution_comparative" />
        <Tidbit variant="bibo" title="BiBo behavior">
          Skywork normalization keeps confidence moderate (0.4–0.7) across all positions.
          This means all 3 selected experts get meaningful weight — the router distributes
          computation rather than concentrating it.
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen behavior">
          Raw softmax can push top-1 confidence to 0.9+ — effectively making it a top-1 router
          that wastes the other K-1 expert computations. The model pays for 3 experts but only
          uses 1 meaningfully.
        </Tidbit>
        <Tidbit variant="insight" title="Why this matters for loss">
          When top_k=3 but confidence is 0.9, the effective expert count is ~1.3 (not 3).
          BiBo&apos;s moderate confidence gives effective count ~2.5 — nearly 2× more expert
          utilization per token. This directly explains the loss gap.
        </Tidbit>
      </div>

      {/* Confidence Distribution */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Confidence Distribution (Violin + Histogram)</h3>
        <SeqTabs prefix="confidence_distribution" />
        <Tidbit variant="neutral" title="Reading the distribution">
          BiBo&apos;s distribution is unimodal and centered — consistent moderate confidence.
          Qwen&apos;s distribution is often bimodal or right-skewed — some tokens get very high
          confidence (wasted experts) while others get moderate.
        </Tidbit>
      </div>

      {/* Entropy */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Routing Entropy Over Position</h3>
        <SeqTabs prefix="entropy_comparative" />
        <Tidbit variant="insight" title="Higher entropy = more uniform weighting">
          Entropy measures how evenly the router distributes weight among selected experts.
          Max entropy for top-3 = log(3) ≈ 1.099. BiBo maintains higher entropy → all selected
          experts get meaningful weight. This is the mechanism behind BiBo&apos;s superior performance.
        </Tidbit>
      </div>

      {/* Weight KDE */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Per-Layer Weight Distribution (KDE)</h3>
        <SeqTabs prefix="weight_kde_per_layer" />
        <Tidbit variant="neutral" title="What the KDE shows">
          Kernel density estimate of routing weights per layer. Peaks near 1/3 (≈0.33) indicate
          uniform distribution among top-3. Peaks near 1.0 indicate one expert dominates.
          BiBo&apos;s peaks cluster around 0.33 — Qwen&apos;s spread wider with a tail toward 1.0.
        </Tidbit>
      </div>
    </div>
  );
}

import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { Tidbit } from '@/components/Tidbit';

export function Confidence() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="↗"
        title="Routing Confidence &amp; Weight Allocation"
        description="How does the router distribute weight among the top-K selected experts? Flat distribution = all experts contribute. Peaked = top-1 dominates and the other K-1 are wasted."
      />

      {/* Confidence Evolution */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Confidence Evolution Over Token Position</h3>
        <SeqTabs prefix="confidence_evolution_comparative" />
        <Tidbit variant="bibo" title="BiBo behavior">
          Router logit normalization keeps confidence moderate (0.4–0.7) across all positions.
          This means all 3 selected experts get meaningful weight — the router distributes
          computation rather than concentrating it.
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen behavior">
          Raw softmax can push top-1 confidence to 0.9+ — effectively making it a top-1 router
          that wastes the other K-1 expert computations. The model pays for 3 experts but only
          uses 1 meaningfully.
        </Tidbit>
      </div>

      {/* Weight KDE — the key plot */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Per-Layer Weight Distribution (KDE)</h3>
        <SeqTabs prefix="weight_kde_per_layer" />
        <Tidbit variant="bibo" title="BiBo: spread distributions across all ranks">
          In BiBo, the rank-1 (blue), rank-2 (orange), and rank-3 (green) weight distributions
          are <strong>spread and overlapping</strong>. Rank-1 peaks around 0.4–0.6, rank-2 around
          0.2–0.4, rank-3 around 0.1–0.3. All three experts get meaningful weight — no single
          expert dominates. This is the direct effect of logit normalization.
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen: rank-1 dominates, others starved">
          In Qwen, rank-1 (blue) peaks near <strong>0.8–1.0</strong> in later layers (L4, L5),
          while rank-2 and rank-3 are pushed toward 0.0–0.2. The router effectively becomes
          top-1 despite selecting top-3. This means 2 of 3 expert computations are wasted —
          the model pays 3× compute for ~1.3× effective expert utilization.
        </Tidbit>
        <Tidbit variant="insight" title="Layer progression tells the story">
          In Qwen, the weight concentration <em>worsens</em> with depth — L1 is relatively balanced
          but by L5, rank-1 gets nearly all weight. The softmax sharpens as the model trains.
          BiBo&apos;s normalization prevents this collapse at every layer, maintaining balanced
          allocation throughout the network depth.
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

      {/* Weight Rank */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Weight Rank Distribution</h3>
        <SeqTabs prefix="weight_rank_distribution" />
        <Tidbit variant="neutral" title="Interpretation">
          Shows average weight given to rank-1, rank-2, rank-3 expert. Flatter = all top-K experts
          contribute equally. BiBo&apos;s router normalization ensures the rank-2 and rank-3 experts
          get meaningful weight, not just crumbs from rank-1.
        </Tidbit>
      </div>
    </div>
  );
}

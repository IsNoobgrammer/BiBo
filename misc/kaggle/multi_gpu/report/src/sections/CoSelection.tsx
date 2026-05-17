import { SectionHeader } from '@/components/SectionHeader';
import { SeqTabs } from '@/components/SeqTabs';
import { Tidbit } from '@/components/Tidbit';

export function CoSelection() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="⊗"
        title="Expert Co-Selection Patterns"
        description="Which expert pairs are frequently selected together for the same token? High co-selection reveals functional clusters — experts that serve complementary roles."
      />

      {/* Co-selection matrices */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Co-Selection Heatmap</h3>
        <SeqTabs prefix="coselection" />
        <Tidbit variant="bibo" title="BiBo: uniform co-selection (0.027–0.046)">
          BiBo&apos;s co-selection matrix is remarkably uniform — all expert pairs are selected
          together at similar rates (range: 0.027–0.046). No pair dominates. This means the
          router explores diverse expert combinations rather than falling into fixed cliques.
          Identity and Zero show slightly elevated co-selection with MLPs, confirming they serve
          complementary roles (cheap processing + full transformation).
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen: extreme favorites (E2↔E7 = 0.162)">
          Qwen&apos;s co-selection is highly non-uniform. The E2↔E7 pair reaches 0.162 — nearly 6×
          the frequency of starved pairs like E1↔E4 (≈0.005). This means 2 of 8 experts form a
          dominant clique that captures most tokens, while others are effectively dead. The aux
          loss can&apos;t break these persistent co-selection patterns.
        </Tidbit>
        <Tidbit variant="insight" title="Why this matters">
          Uniform co-selection = all expert combinations are explored = richer representations.
          Skewed co-selection = the model wastes capacity on redundant expert pairs while starving
          others. BiBo&apos;s logit normalization prevents any pair from dominating, while Qwen&apos;s
          raw softmax amplifies initial preferences into permanent cliques.
        </Tidbit>
      </div>

      {/* Specialization Radar */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Specialization Radar</h3>
        <p className="text-xs text-white/50 mb-3">
          Per-expert position specialization measured as KL divergence from uniform distribution.
          Higher = expert strongly prefers specific token positions.
        </p>
        <SeqTabs prefix="specialization_radar" />
        <Tidbit variant="bibo" title="BiBo specialization">
          Special experts (Identity, Zero) show higher specialization — they&apos;re
          activated for specific token types. Identity handles unsorted input tokens
          (already-good representations), while ReLU² activates more on sorted output
          tokens (needing sharp feature selection for generation).
        </Tidbit>
        <Tidbit variant="neutral" title="Interpretation">
          Some specialization is good — it means experts have learned distinct roles.
          Too much specialization (one expert only handles position 0) would indicate
          overfitting to position rather than content. Moderate specialization is ideal.
        </Tidbit>
      </div>
    </div>
  );
}

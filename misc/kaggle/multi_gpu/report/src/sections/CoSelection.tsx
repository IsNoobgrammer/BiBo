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
        <Tidbit variant="bibo" title="BiBo patterns">
          Look for MLP + Identity pairs — the MLP transforms while Identity preserves.
          Also look for MLP + Zero pairs — the MLP handles complex tokens while Zero
          delegates to the shared Conv1D for simple ones. These complementary pairs
          show the router has learned functional specialization.
        </Tidbit>
        <Tidbit variant="qwen" title="Qwen patterns">
          With homogeneous experts, co-selection patterns are less meaningful — all experts
          do the same operation (SwiGLU). High co-selection just means &quot;these two experts
          happen to have similar weight vectors,&quot; not functional complementarity.
        </Tidbit>
        <Tidbit variant="insight" title="What to look for">
          Diagonal = self-selection (always 1.0). Off-diagonal values near 0.5+ indicate
          strong pairing. In BiBo, the special experts (Identity, Zero, ReLU²) should show
          distinct co-selection patterns from the MLP experts.
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
          Special experts (Identity, Zero) tend to show higher specialization — they&apos;re
          activated for specific token types (e.g., Zero for padding-like tokens, Identity
          for already-well-represented tokens). MLP experts are more general-purpose.
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

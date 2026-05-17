import { MetricCard } from '@/components/MetricCard';
import { Tidbit } from '@/components/Tidbit';
import { PlotImage } from '@/components/PlotImage';

export function Overview() {
  return (
    <div className="space-y-8">
      {/* Key Finding */}
      <div className="glass rounded-2xl p-6 md:p-8 border border-emerald-500/20 bg-emerald-500/[0.03]">
        <div className="flex items-start gap-3 mb-4">
          <span className="text-xl">⚡</span>
          <div>
            <h2 className="text-lg font-bold text-white mb-1">Key Finding</h2>
            <p className="text-white/60 text-sm leading-relaxed">
              BiBo achieves <strong className="text-emerald-400">lower validation loss (0.0039 vs 0.0045), better load balance, AND superior generalization</strong> compared to Qwen3MoE.
              At seq256/bs64, BiBo&apos;s router logit normalization + bias heuristics produce near-perfect balance
              (Gini=0.022, ratio=0.90) while Qwen develops extreme imbalance (Gini=0.235, ratio=0.24).
              BiBo wins 3-1 on unseen sequence lengths, with better confidence calibration across the board.
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-white/40 mb-4">Head-to-Head at seq=256, bs=64</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="H_norm" value="0.9996" variant="bibo" winner subtitle="Normalized entropy" />
          <MetricCard label="H_norm" value="0.9564" variant="qwen" subtitle="Normalized entropy" />
          <MetricCard label="Gini" value="0.022" variant="bibo" winner subtitle="Lower = more balanced" />
          <MetricCard label="Gini" value="0.235" variant="qwen" subtitle="Lower = more balanced" />
          <MetricCard label="Balance Ratio" value="0.90" variant="bibo" winner subtitle="min/max load" />
          <MetricCard label="Balance Ratio" value="0.24" variant="qwen" subtitle="min/max load" />
          <MetricCard label="Val Loss" value="0.0039" variant="bibo" winner subtitle="14% lower" />
          <MetricCard label="Val Loss" value="0.0045" variant="qwen" subtitle="Sorting task" />
        </div>
      </div>

      {/* Generalization Score */}
      <div className="glass rounded-xl p-5 border border-bibo-500/20 bg-bibo-500/[0.02]">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Length Generalization — Score: BiBo 3 — Qwen 1</h3>
        <PlotImage src="model_length_generalization" alt="Length Generalization" />
        <Tidbit variant="insight" title="Generalization test">
          Tested on sequence lengths NOT in training data (8, 96, 128, 192). BiBo wins 3 out of 4,
          demonstrating that SSMax + conv router + diverse experts generalize better to unseen lengths.
          Qwen only wins at seq=96 (interpolation between training lengths 64 and 128).
        </Tidbit>
      </div>

      {/* Architecture Comparison */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-bibo-400 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-bibo-400" />
            BiBo Router
          </h3>
          <ul className="space-y-2 text-xs text-white/60">
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Conv1D router (kernel=3) — sees local context</li>
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Router logit normalization (λ=1.0)</li>
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Bias heuristics (threshold-based updates)</li>
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Heterogeneous experts: MLP + Identity + Zero + ReLU²</li>
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Always-active shared CausalConv1D expert</li>
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> No auxiliary loss — routing is self-balancing</li>
          </ul>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="text-sm font-bold text-qwen-400 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-qwen-400" />
            Qwen3MoE Router
          </h3>
          <ul className="space-y-2 text-xs text-white/60">
            <li className="flex items-start gap-2"><span className="text-qwen-400/60 mt-0.5">▸</span> Linear router — each token routed independently</li>
            <li className="flex items-start gap-2"><span className="text-qwen-400/60 mt-0.5">▸</span> Raw softmax over logits</li>
            <li className="flex items-start gap-2"><span className="text-qwen-400/60 mt-0.5">▸</span> Auxiliary load-balancing loss</li>
            <li className="flex items-start gap-2"><span className="text-qwen-400/60 mt-0.5">▸</span> Homogeneous experts: all SwiGLU MLPs</li>
            <li className="flex items-start gap-2"><span className="text-qwen-400/60 mt-0.5">▸</span> Shared expert (same architecture as routed)</li>
            <li className="flex items-start gap-2"><span className="text-qwen-400/60 mt-0.5">▸</span> Proven at scale in production</li>
          </ul>
        </div>
      </div>

      {/* Grand Summary Plot */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Grand Summary Dashboard</h3>
        <PlotImage src="grand_summary_dashboard" alt="Grand Summary Dashboard" />
        <Tidbit variant="insight" title="Reading this plot">
          Each subplot compares a different routing metric across all configurations.
          BiBo (blue) consistently outperforms on entropy, balance ratio, and effective expert count,
          while maintaining lower CV (coefficient of variation). The gap is dramatic — Qwen&apos;s
          Gini is 10× worse and balance ratio is 4× worse.
        </Tidbit>
      </div>

      {/* Model Quality Comparison */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Model Quality Comparison</h3>
        <PlotImage src="model_quality_comparison" alt="Model Quality Comparison" />
      </div>

      {/* Confidence Position */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Confidence by Position</h3>
        <PlotImage src="model_confidence_position" alt="Model Confidence by Position" />
        <Tidbit variant="insight" title="Confidence calibration">
          BiBo shows larger gap between correct and incorrect prediction confidence across all
          sequence lengths — better calibrated uncertainty. At seq=192 (unseen), BiBo&apos;s gap is
          0.145 vs Qwen&apos;s 0.073, meaning BiBo &quot;knows when it doesn&apos;t know.&quot;
        </Tidbit>
      </div>

      {/* Quick Stats */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Why BiBo Wins: The Numbers</h3>
        <div className="grid md:grid-cols-3 gap-4 text-xs text-white/60">
          <div>
            <p className="font-mono text-bibo-400 text-lg mb-1">10×</p>
            <p>Better Gini coefficient at seq256 (0.022 vs 0.235)</p>
          </div>
          <div>
            <p className="font-mono text-bibo-400 text-lg mb-1">3.75×</p>
            <p>Higher balance ratio (0.90 vs 0.24) — min expert gets far more tokens</p>
          </div>
          <div>
            <p className="font-mono text-bibo-400 text-lg mb-1">3–1</p>
            <p>Generalization score on unseen sequence lengths</p>
          </div>
        </div>
      </div>
    </div>
  );
}

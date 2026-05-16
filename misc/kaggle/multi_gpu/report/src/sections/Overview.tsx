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
              BiBo achieves <strong className="text-emerald-400">both better load balance AND lower loss</strong> than Qwen3MoE.
              At seq128, BiBo&apos;s Skywork normalization + heuristic bias updates produce near-perfect balance
              (Gini=0.019, ratio=0.88) while Qwen&apos;s aux loss results in worse balance (Gini=0.076, ratio=0.68).
              Combined with 2.5× lower final loss and 26% fewer parameters, BiBo&apos;s routing is strictly superior on this task.
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-white/40 mb-4">Head-to-Head at seq=128, bs=64</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="H_norm" value="0.9997" variant="bibo" winner subtitle="Normalized entropy" />
          <MetricCard label="H_norm" value="0.9956" variant="qwen" subtitle="Normalized entropy" />
          <MetricCard label="Gini" value="0.019" variant="bibo" winner subtitle="Lower = more balanced" />
          <MetricCard label="Gini" value="0.076" variant="qwen" subtitle="Lower = more balanced" />
          <MetricCard label="Balance Ratio" value="0.88" variant="bibo" winner subtitle="min/max load" />
          <MetricCard label="Balance Ratio" value="0.68" variant="qwen" subtitle="min/max load" />
          <MetricCard label="Final Loss" value="0.10" variant="bibo" winner subtitle="2.5× lower" />
          <MetricCard label="Final Loss" value="0.25" variant="qwen" subtitle="Sorting task" />
        </div>
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
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Skywork-MoE logit normalization (λ=1.0)</li>
            <li className="flex items-start gap-2"><span className="text-bibo-400/60 mt-0.5">▸</span> Heuristic bias updates (threshold-based)</li>
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
          while maintaining lower CV (coefficient of variation). The gap widens at longer sequences.
        </Tidbit>
      </div>

      {/* Quick Stats */}
      <div className="glass rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white/80 mb-3">Why BiBo Wins: The Numbers</h3>
        <div className="grid md:grid-cols-3 gap-4 text-xs text-white/60">
          <div>
            <p className="font-mono text-bibo-400 text-lg mb-1">4×</p>
            <p>Better Gini coefficient at seq128 (0.019 vs 0.076)</p>
          </div>
          <div>
            <p className="font-mono text-bibo-400 text-lg mb-1">2.5×</p>
            <p>Lower final loss (0.10 vs 0.25) with 26% fewer parameters</p>
          </div>
          <div>
            <p className="font-mono text-bibo-400 text-lg mb-1">29%</p>
            <p>Higher balance ratio (0.88 vs 0.68) — min expert gets more tokens</p>
          </div>
        </div>
      </div>
    </div>
  );
}

import { SectionHeader } from '@/components/SectionHeader';
import { Tidbit } from '@/components/Tidbit';

export function Verdict() {
  return (
    <div className="space-y-8">
      <SectionHeader
        icon="⚖"
        title="Final Verdict"
        description="A comprehensive comparison of both routing strategies, their strengths, weaknesses, and ideal use cases."
      />

      {/* Verdict Cards */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* BiBo */}
        <div className="glass rounded-xl p-6 border-t-2 border-t-bibo-500">
          <h3 className="text-lg font-bold text-bibo-400 mb-4">BiBo Router</h3>
          <div className="space-y-4">
            <div className="bg-emerald-500/5 rounded-lg p-4 border border-emerald-500/10">
              <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">Strengths</h4>
              <ul className="space-y-1.5 text-xs text-white/60">
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Context-aware routing (Conv1D, kernel=3)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Meaningful expert diversity (Identity, Zero, ReLU²)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Router logit normalization prevents expert collapse</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> No auxiliary loss needed — self-balancing</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Near-perfect balance at seq256 (Gini=0.022, ratio=0.90)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> 14% lower validation loss (0.0039 vs 0.0045)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Wins 3-1 on unseen sequence length generalization</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Better confidence calibration (knows when it&apos;s wrong)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> ~35% compute savings from cheap experts</li>
              </ul>
            </div>
            <div className="bg-red-500/5 rounded-lg p-4 border border-red-500/10">
              <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider mb-2">Weaknesses</h4>
              <ul className="space-y-1.5 text-xs text-white/60">
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Lower stability (0.39) — context-dependent, harder to debug</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Conv incompatible with sequence parallelism</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Novel — less battle-tested at scale</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Slightly lower val accuracy (0.9981 vs 0.9987)</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Qwen */}
        <div className="glass rounded-xl p-6 border-t-2 border-t-qwen-500">
          <h3 className="text-lg font-bold text-qwen-400 mb-4">Qwen3MoE Router</h3>
          <div className="space-y-4">
            <div className="bg-emerald-500/5 rounded-lg p-4 border border-emerald-500/10">
              <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2">Strengths</h4>
              <ul className="space-y-1.5 text-xs text-white/60">
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Very high stability (0.93) — deterministic, debuggable</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Simple architecture — easy to parallelize</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Proven at scale in production (Qwen series)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Slightly higher val accuracy (0.9987 vs 0.9981)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Compatible with all parallelism strategies</li>
              </ul>
            </div>
            <div className="bg-red-500/5 rounded-lg p-4 border border-red-500/10">
              <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider mb-2">Weaknesses</h4>
              <ul className="space-y-1.5 text-xs text-white/60">
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> No context awareness — each token isolated</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Homogeneous experts only (all SwiGLU)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Extreme load imbalance (Gini=0.235, ratio=0.24)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Higher validation loss (0.0045 vs 0.0039)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Loses 1-3 on generalization to unseen lengths</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Worse confidence calibration (overconfident on errors)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> No &quot;cheap&quot; expert option — all tokens pay full cost</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Persistent favorites (4× load imbalance)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* When to Use */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-4">When to Use Each</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-3 px-3 text-white/40 uppercase tracking-wider font-semibold">Scenario</th>
                <th className="text-left py-3 px-3 text-white/40 uppercase tracking-wider font-semibold">Winner</th>
                <th className="text-left py-3 px-3 text-white/40 uppercase tracking-wider font-semibold">Why</th>
              </tr>
            </thead>
            <tbody className="text-white/60">
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Expert Parallelism (64+ experts)</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Even load = no stragglers; Identity/Zero shards need no GPU compute</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Single GPU (≤16 experts)</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">10× better balance + diversity + lower loss</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Length generalization</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Wins 3-1 on unseen lengths; SSMax + conv router</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Production (proven track record)</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-qwen-500/10 text-qwen-400 font-medium">Qwen</span></td>
                <td className="py-3 px-3">Battle-tested at scale; BiBo is unproven beyond research</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Research / novel architectures</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Heterogeneous experts explore the design space</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Compute-constrained</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Zero/Identity = free skip connections, ~35% MoE compute savings</td>
              </tr>
              <tr className="hover:bg-white/[0.02]">
                <td className="py-3 px-3">Load balance critical</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Gini=0.022 vs Qwen&apos;s 0.235 — 10× better</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Final Insight */}
      <div className="glass rounded-xl p-6 border border-bibo-500/20 bg-bibo-500/[0.02]">
        <Tidbit variant="insight" title="Bottom line">
          On this sorting task with 8 experts and top-3 routing, BiBo dominates on load balance
          (10× better Gini), generalization (3-1 on unseen lengths), validation loss (14% lower),
          and confidence calibration. Qwen&apos;s extreme load imbalance (ratio=0.24) means 4 of 8
          experts are effectively starved. Qwen&apos;s advantages are operational (near-deterministic
          routing, parallelizability, proven at scale) — important for production but not for
          model capability. The question isn&apos;t &quot;which is better&quot; — it&apos;s &quot;what do you optimize for.&quot;
        </Tidbit>
      </div>
    </div>
  );
}

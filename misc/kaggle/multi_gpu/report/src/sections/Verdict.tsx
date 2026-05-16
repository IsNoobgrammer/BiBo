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
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Better balance at seq128+ (Gini=0.019)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Balance ratio 0.88 (vs Qwen&apos;s 0.68)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> 2.5× lower final loss with 26% fewer params</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> ~35% compute savings from cheap experts</li>
              </ul>
            </div>
            <div className="bg-red-500/5 rounded-lg p-4 border border-red-500/10">
              <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider mb-2">Weaknesses</h4>
              <ul className="space-y-1.5 text-xs text-white/60">
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Lower stability (0.31) — harder to debug</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Conv incompatible with sequence parallelism</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Bias heuristics need tokens to converge (seq64 weaker)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Novel — less battle-tested at scale</li>
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
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> High stability (0.47) — predictable, debuggable</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Simple architecture — easy to parallelize</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Proven at scale in production (Qwen series)</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Continuous gradient signal via aux loss</li>
                <li className="flex items-start gap-2"><span className="text-emerald-400">+</span> Compatible with all parallelism strategies</li>
              </ul>
            </div>
            <div className="bg-red-500/5 rounded-lg p-4 border border-red-500/10">
              <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider mb-2">Weaknesses</h4>
              <ul className="space-y-1.5 text-xs text-white/60">
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> No context awareness — each token isolated</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Homogeneous experts only (all SwiGLU)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Aux loss interferes with task loss gradient</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Worse balance than BiBo (Gini=0.076)</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Higher final loss (0.25) with more params</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> No &quot;cheap&quot; expert option — all tokens pay full cost</li>
                <li className="flex items-start gap-2"><span className="text-red-400">−</span> Persistent favorites (balance_ratio=0.68)</li>
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
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-qwen-500/10 text-qwen-400 font-medium">Qwen</span></td>
                <td className="py-3 px-3">Predictable routing, simpler to shard across nodes</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Single GPU (≤16 experts)</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Better balance + diversity + lower loss + fewer params</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Long context tasks</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-bibo-500/10 text-bibo-400 font-medium">BiBo</span></td>
                <td className="py-3 px-3">Conv router + SSMax; balance improves with seq length</td>
              </tr>
              <tr className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="py-3 px-3">Production deployment</td>
                <td className="py-3 px-3"><span className="px-2 py-0.5 rounded bg-qwen-500/10 text-qwen-400 font-medium">Qwen</span></td>
                <td className="py-3 px-3">Higher stability (0.47), debuggable, battle-tested</td>
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
                <td className="py-3 px-3">Gini=0.019 at seq128 vs Qwen&apos;s 0.076</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Final Insight */}
      <div className="glass rounded-xl p-6 border border-bibo-500/20 bg-bibo-500/[0.02]">
        <Tidbit variant="insight" title="Bottom line">
          On this sorting task with 8 experts and top-3 routing, BiBo dominates across every metric
          that matters for model quality: lower loss, better balance, higher effective expert utilization,
          and context-aware routing. Qwen&apos;s advantages are operational (stability, parallelizability,
          proven at scale) — important for production but not for model capability.
          The question isn&apos;t &quot;which is better&quot; — it&apos;s &quot;what do you optimize for.&quot;
        </Tidbit>
      </div>
    </div>
  );
}

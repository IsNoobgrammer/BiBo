'use client';

import { useState } from 'react';
import { sections } from '@/data/metrics';
import { Overview } from '@/sections/Overview';
import { Balance } from '@/sections/Balance';
import { Confidence } from '@/sections/Confidence';
import { Heatmaps } from '@/sections/Heatmaps';
import { CoSelection } from '@/sections/CoSelection';
import { Stability } from '@/sections/Stability';
import { Switching } from '@/sections/Switching';
import { SpecialExperts } from '@/sections/SpecialExperts';
import { Verdict } from '@/sections/Verdict';

export default function Home() {
  const [activeSection, setActiveSection] = useState('overview');

  const renderSection = () => {
    switch (activeSection) {
      case 'overview': return <Overview />;
      case 'balance': return <Balance />;
      case 'confidence': return <Confidence />;
      case 'heatmaps': return <Heatmaps />;
      case 'coselection': return <CoSelection />;
      case 'stability': return <Stability />;
      case 'switching': return <Switching />;
      case 'special': return <SpecialExperts />;
      case 'verdict': return <Verdict />;
      default: return <Overview />;
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-bibo-900/40 via-surface-0 to-qwen-900/30" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(59,147,246,0.08),transparent_50%)]" />
        <div className="relative px-6 pt-16 pb-12 text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs text-white/60 mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Research Report — May 2026
          </div>
          <h1 className="text-4xl md:text-5xl font-black tracking-tight mb-3">
            <span className="text-bibo-400">BiBo</span>
            <span className="text-white/30 mx-3 font-light">vs</span>
            <span className="text-qwen-400">Qwen3MoE</span>
          </h1>
          <p className="text-lg text-white/50 font-light mb-6">
            Comprehensive MoE Router Behavior Analysis
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {['8 Experts', 'Top-3 Routing', 'Seq 64 / 128 / 256', 'BS 1→64', 'Sorting Task', '2×T4 GPUs'].map(badge => (
              <span key={badge} className="px-3 py-1 rounded-full bg-white/5 border border-white/8 text-xs text-white/50 font-medium">
                {badge}
              </span>
            ))}
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 py-2.5 flex gap-1 overflow-x-auto scrollbar-hide">
          {sections.map(s => (
            <button
              key={s.id}
              onClick={() => setActiveSection(s.id)}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-150 ${
                activeSection === s.id
                  ? 'bg-white/10 text-white border border-white/15'
                  : 'text-white/40 hover:text-white/70 hover:bg-white/5 border border-transparent'
              }`}
            >
              <span className="text-[10px]">{s.icon}</span>
              {s.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        <div className="animate-slide-up">
          {renderSection()}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 text-center">
        <p className="text-xs text-white/30">
          BiBo Router Analysis • Trained on sorting task • 2×T4 16GB •{' '}
          <a href="https://github.com/IsNoobgrammer/BiBo" className="text-bibo-400/60 hover:text-bibo-400 transition-colors">
            github.com/IsNoobgrammer/BiBo
          </a>
        </p>
      </footer>
    </div>
  );
}

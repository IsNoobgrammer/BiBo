'use client';

import { useState } from 'react';
import { PlotImage } from './PlotImage';
import type { SeqLen } from '@/data/metrics';

interface SeqTabsProps {
  prefix: string;
  suffix?: string;
  seqLens?: SeqLen[];
}

export function SeqTabs({ prefix, suffix = '', seqLens = [64, 128, 256] }: SeqTabsProps) {
  const [active, setActive] = useState<SeqLen>(seqLens[0]);

  const getFilename = (seq: SeqLen) => {
    return suffix ? `${prefix}_seq${seq}_${suffix}` : `${prefix}_seq${seq}`;
  };

  return (
    <div>
      <div className="flex gap-1.5 mb-3">
        {seqLens.map(seq => (
          <button
            key={seq}
            onClick={() => setActive(seq)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 ${
              active === seq
                ? 'bg-white/10 text-white border border-white/20'
                : 'text-white/40 hover:text-white/70 border border-transparent'
            }`}
          >
            seq={seq}
          </button>
        ))}
      </div>
      <PlotImage src={getFilename(active)} alt={`${prefix} seq=${active}`} />
    </div>
  );
}

export const config = {
  experts: 8,
  topK: 3,
  seqLens: [64, 128, 256] as const,
  batchSizes: [1, 5, 20, 64] as const,
};

export const keyMetrics = {
  bibo: {
    seq128_bs64: { entropy: 2.0788, h_norm: 0.9997, gini: 0.019, cv: 0.036, balance_ratio: 0.88 },
    seq64_bs64: { entropy: 2.0712, h_norm: 0.996, gini: 0.07, cv: 0.13, balance_ratio: 0.65 },
    seq256_bs64: { entropy: 2.0768, h_norm: 0.9987, gini: 0.037, cv: 0.072, balance_ratio: 0.77 },
    finalLoss: 0.1,
    params: '8.3M',
  },
  qwen: {
    seq128_bs64: { entropy: 2.0703, h_norm: 0.9956, gini: 0.076, cv: 0.136, balance_ratio: 0.68 },
    seq64_bs64: { entropy: 2.0682, h_norm: 0.9946, gini: 0.084, cv: 0.15, balance_ratio: 0.67 },
    seq256_bs64: { entropy: 2.0703, h_norm: 0.9956, gini: 0.076, cv: 0.135, balance_ratio: 0.67 },
    finalLoss: 0.25,
    params: '11.2M',
  },
};

export const stabilityScores = {
  bibo: { seq64: 0.31, seq128: 0.30, seq256: 0.32, avg: 0.31 },
  qwen: { seq64: 0.44, seq128: 0.47, seq256: 0.49, avg: 0.47 },
};

export type SeqLen = 64 | 128 | 256;

export const sections = [
  { id: 'overview', label: 'Overview', icon: '◉' },
  { id: 'balance', label: 'Load Balance', icon: '⊞' },
  { id: 'confidence', label: 'Confidence', icon: '↗' },
  { id: 'heatmaps', label: 'Heatmaps', icon: '▦' },
  { id: 'coselection', label: 'Co-Selection', icon: '⊗' },
  { id: 'stability', label: 'Stability', icon: '◎' },
  { id: 'switching', label: 'Switching', icon: '⇄' },
  { id: 'special', label: 'Special Experts', icon: '✦' },
  { id: 'verdict', label: 'Verdict', icon: '⚖' },
] as const;

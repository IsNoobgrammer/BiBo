export const config = {
  experts: 8,
  topK: 3,
  seqLens: [64, 256] as const,
  batchSizes: [1, 5, 20, 64] as const,
};

export const keyMetrics = {
  bibo: {
    seq64_bs64: { entropy: 2.0779, h_norm: 0.9993, gini: 0.029, cv: 0.055, balance_ratio: 0.83 },
    seq256_bs64: { entropy: 2.0786, h_norm: 0.9996, gini: 0.022, cv: 0.041, balance_ratio: 0.90 },
    valLoss: 0.003894,
    valAcc: 0.9981,
    trainLoss: 0.004425,
    params: '~10M',
  },
  qwen: {
    seq64_bs64: { entropy: 1.9777, h_norm: 0.9511, gini: 0.249, cv: 0.447, balance_ratio: 0.20 },
    seq256_bs64: { entropy: 1.9889, h_norm: 0.9564, gini: 0.235, cv: 0.425, balance_ratio: 0.24 },
    valLoss: 0.004459,
    valAcc: 0.9987,
    trainLoss: 0.004361,
    params: '~10M',
  },
};

export const stabilityScores = {
  bibo: { seq64: 0.391, seq256: 0.392, avg: 0.391 },
  qwen: { seq64: 0.929, seq256: 0.934, avg: 0.931 },
};

export const generalization = {
  // Sequence lengths NOT in training data marked with *
  results: [
    { seq: 8, inTrain: false, bibo: { loss: 10.5931, acc: 0.9531, fullSeq: 0.625 }, qwen: { loss: 10.794, acc: 0.9375, fullSeq: 0.75 }, winner: 'BiBo' },
    { seq: 32, inTrain: true, bibo: { loss: 10.928, acc: 1.0, fullSeq: 1.0 }, qwen: { loss: 9.7074, acc: 1.0, fullSeq: 1.0 }, winner: 'Tie' },
    { seq: 64, inTrain: true, bibo: { loss: 10.0956, acc: 1.0, fullSeq: 1.0 }, qwen: { loss: 9.0234, acc: 1.0, fullSeq: 1.0 }, winner: 'Tie' },
    { seq: 96, inTrain: false, bibo: { loss: 8.3743, acc: 0.9766, fullSeq: 0.0 }, qwen: { loss: 8.0889, acc: 0.9896, fullSeq: 0.5 }, winner: 'Qwen' },
    { seq: 128, inTrain: false, bibo: { loss: 7.6232, acc: 0.9561, fullSeq: 0.0 }, qwen: { loss: 7.7074, acc: 0.9229, fullSeq: 0.0 }, winner: 'BiBo' },
    { seq: 192, inTrain: false, bibo: { loss: 7.1916, acc: 0.9596, fullSeq: 0.0 }, qwen: { loss: 6.8793, acc: 0.9355, fullSeq: 0.0 }, winner: 'BiBo' },
  ],
  score: { bibo: 3, qwen: 1 },
};

export const confidence = {
  results: [
    { seq: 8, bibo: { correct: 0.9781, incorrect: 0.7534, gap: 0.2247 }, qwen: { correct: 0.9699, incorrect: 0.8004, gap: 0.1695 } },
    { seq: 32, bibo: { correct: 0.9996, incorrect: 0.0, gap: 0.9996 }, qwen: { correct: 0.998, incorrect: 0.0, gap: 0.998 } },
    { seq: 64, bibo: { correct: 0.999, incorrect: 0.0, gap: 0.999 }, qwen: { correct: 0.9987, incorrect: 0.0, gap: 0.9987 } },
    { seq: 96, bibo: { correct: 0.9926, incorrect: 0.8694, gap: 0.1232 }, qwen: { correct: 0.9912, incorrect: 0.8739, gap: 0.1172 } },
    { seq: 128, bibo: { correct: 0.9886, incorrect: 0.8709, gap: 0.1176 }, qwen: { correct: 0.9807, incorrect: 0.8782, gap: 0.1025 } },
    { seq: 192, bibo: { correct: 0.9908, incorrect: 0.846, gap: 0.1448 }, qwen: { correct: 0.979, incorrect: 0.9064, gap: 0.0726 } },
  ],
};

export type SeqLen = 64 | 256;

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

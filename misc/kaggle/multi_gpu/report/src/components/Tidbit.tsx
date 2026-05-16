interface TidbitProps {
  variant?: 'bibo' | 'qwen' | 'neutral' | 'insight';
  title?: string;
  children: React.ReactNode;
}

export function Tidbit({ variant = 'neutral', title, children }: TidbitProps) {
  const borderColors = {
    bibo: 'border-l-bibo-500/50',
    qwen: 'border-l-qwen-500/50',
    neutral: 'border-l-white/20',
    insight: 'border-l-emerald-500/50',
  };

  const bgColors = {
    bibo: 'bg-bibo-500/5',
    qwen: 'bg-qwen-500/5',
    neutral: 'bg-white/[0.02]',
    insight: 'bg-emerald-500/5',
  };

  const icons = {
    bibo: '◆',
    qwen: '◆',
    neutral: '→',
    insight: '💡',
  };

  return (
    <div className={`border-l-2 ${borderColors[variant]} ${bgColors[variant]} pl-4 py-3 pr-4 rounded-r-lg mt-3`}>
      {title && (
        <p className="text-xs font-semibold uppercase tracking-wider text-white/50 mb-1">
          {icons[variant]} {title}
        </p>
      )}
      <p className="text-sm text-white/70 leading-relaxed">{children}</p>
    </div>
  );
}

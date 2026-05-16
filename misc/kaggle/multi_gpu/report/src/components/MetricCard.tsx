interface MetricCardProps {
  label: string;
  value: string | number;
  variant: 'bibo' | 'qwen';
  winner?: boolean;
  subtitle?: string;
}

export function MetricCard({ label, value, variant, winner = false, subtitle }: MetricCardProps) {
  return (
    <div className={`relative rounded-xl p-4 ${
      variant === 'bibo' ? 'metric-glow-bibo' : 'metric-glow-qwen'
    } ${winner ? 'ring-1 ring-emerald-500/30 bg-emerald-500/5' : 'bg-white/[0.02]'}`}>
      {winner && (
        <div className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-emerald-500 rounded-full flex items-center justify-center">
          <span className="text-[9px] font-bold text-white">✓</span>
        </div>
      )}
      <p className={`text-2xl font-bold tracking-tight ${
        variant === 'bibo' ? 'text-bibo-400' : 'text-qwen-400'
      }`}>
        {value}
      </p>
      <p className="text-[11px] font-medium uppercase tracking-wider text-white/40 mt-1">{label}</p>
      {subtitle && <p className="text-[10px] text-white/30 mt-0.5">{subtitle}</p>}
    </div>
  );
}

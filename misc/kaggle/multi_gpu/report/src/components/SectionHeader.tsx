interface SectionHeaderProps {
  icon: string;
  title: string;
  description: string;
}

export function SectionHeader({ icon, title, description }: SectionHeaderProps) {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-2xl">{icon}</span>
        <h2 className="text-2xl font-bold tracking-tight text-white">{title}</h2>
      </div>
      <p className="text-white/50 text-sm leading-relaxed max-w-2xl">{description}</p>
    </div>
  );
}

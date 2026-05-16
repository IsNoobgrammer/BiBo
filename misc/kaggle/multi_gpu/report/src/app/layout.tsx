import type { Metadata } from 'next';
import './globals.css';
import { ThemeToggle } from '@/components/ThemeToggle';

export const metadata: Metadata = {
  title: 'BiBo vs Qwen3MoE — Router Analysis',
  description: 'Comprehensive MoE router behavior analysis comparing BiBo (logit norm + Conv router + bias heuristics) vs Qwen3MoE (aux loss + linear router)',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-surface-0 text-white antialiased font-sans transition-colors duration-300">
        <ThemeToggle />
        {children}
      </body>
    </html>
  );
}

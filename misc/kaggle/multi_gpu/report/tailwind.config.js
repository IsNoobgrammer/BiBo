/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bibo: { 50: '#eff8ff', 100: '#dbeffe', 200: '#bfe3fe', 300: '#93d1fd', 400: '#60b5fa', 500: '#3b93f6', 600: '#2574eb', 700: '#1d5dd8', 800: '#1e4caf', 900: '#1e428a' },
        qwen: { 50: '#fff7ed', 100: '#ffedd5', 200: '#fed7aa', 300: '#fdba74', 400: '#fb923c', 500: '#f97316', 600: '#ea580c', 700: '#c2410c', 800: '#9a3412', 900: '#7c2d12' },
        surface: { 0: '#0a0a0f', 1: '#12121a', 2: '#1a1a24', 3: '#22222e', 4: '#2a2a38' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};

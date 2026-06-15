import { Github, FileText } from 'lucide-react';

const links = [
  { href: '#key-results',   label: 'Results' },
  { href: '#why-text-fails', label: 'Motivation' },
  { href: '#method',        label: 'Method' },
  { href: '#path-a',        label: 'Path A' },
  { href: '#adversarial',   label: 'Adversarial' },
  { href: '#cross-family',  label: 'LM family' },
  { href: '#vlm',           label: 'VLM' },
  { href: '#demo',          label: 'Demo' },
  { href: '#reproducibility', label: 'Reproduce' },
];

export function Navbar() {
  return (
    <header className="sticky top-0 z-40 bg-paper/90 backdrop-blur border-b border-gray-200">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between gap-4">
        <a href="#top" className="font-serif font-bold text-lg tracking-tight text-brand-dark">
          Parametric Multimodal User Memory
        </a>
        <nav className="hidden md:flex items-center gap-5 text-sm text-gray-700">
          {links.map(l => (
            <a key={l.href} href={l.href} className="hover:text-brand transition-colors">
              {l.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <a
            href="https://github.com/bojieli/multimodal-user-memory"
            target="_blank" rel="noopener"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-ink text-paper text-sm font-medium hover:bg-brand-dark transition-colors"
          >
            <Github size={14} /> Code
          </a>
          <a
            href={`${import.meta.env.BASE_URL}main.pdf`}
            target="_blank"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-gray-300 text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            <FileText size={14} /> Paper
          </a>
        </div>
      </div>
    </header>
  );
}

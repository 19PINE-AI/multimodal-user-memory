import { ExternalLink, Github, FileText, Mail } from 'lucide-react';
import { PAPER } from '../paper';

export function Footer() {
  return (
    <footer className="bg-brand-dark text-paper py-12 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          <div>
            <div className="font-serif font-bold text-lg mb-3">{PAPER.title}</div>
            <p className="text-sm text-gray-300 leading-relaxed">
              A parametric multimodal user memory for LM agents — storing what captions cannot carry.
            </p>
          </div>
          <div>
            <h4 className="font-bold text-sm uppercase tracking-wider mb-3">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li><a href={PAPER.abstractUrl} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><ExternalLink size={14} /> arXiv abstract</a></li>
              <li><a href={PAPER.pdfUrl} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><FileText size={14} /> Paper PDF</a></li>
              <li><a href={PAPER.repositoryUrl} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><Github size={14} /> GitHub repository</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-sm uppercase tracking-wider mb-3">Contact</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="mailto:boj@19pine.ai" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><Mail size={14} /> boj@19pine.ai</a></li>
              <li className="text-gray-400">Pine AI</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-brand-light/30 pt-6 text-xs text-gray-400 text-center">
          Companion site to the paper “{PAPER.title}: {PAPER.subtitle}” · arXiv:{PAPER.arxivId}.
          Built with React + Vite + Tailwind + Recharts.
        </div>
      </div>
    </footer>
  );
}

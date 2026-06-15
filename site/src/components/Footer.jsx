import { Github, FileText, Mail } from 'lucide-react';

export function Footer() {
  return (
    <footer className="bg-brand-dark text-paper py-12 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          <div>
            <div className="font-serif font-bold text-lg mb-3">Perceptual Engram</div>
            <p className="text-sm text-gray-300 leading-relaxed">
              A parametric multimodal user memory for LM agents — storing what captions cannot carry.
            </p>
          </div>
          <div>
            <h4 className="font-bold text-sm uppercase tracking-wider mb-3">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li><a href={`${import.meta.env.BASE_URL}main.pdf`} target="_blank" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><FileText size={14} /> Paper PDF</a></li>
              <li><a href="https://github.com/bojieli/multimodal-user-memory" target="_blank" rel="noopener" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><Github size={14} /> GitHub repository</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-sm uppercase tracking-wider mb-3">Contact</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="mailto:bojieli@gmail.com" className="text-gray-300 hover:text-paper flex items-center gap-1.5"><Mail size={14} /> bojieli@gmail.com</a></li>
              <li className="text-gray-400">Pine AI</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-brand-light/30 pt-6 text-xs text-gray-400 text-center">
          Companion site to the paper "Parametric Multimodal User Memory: Storing What Captions Cannot Carry".
          Built with React + Vite + Tailwind + Recharts.
        </div>
      </div>
    </footer>
  );
}

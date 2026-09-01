import { ExternalLink, Github } from 'lucide-react';
import { PAPER } from '../paper';

export function Reproducibility() {
  return (
    <section id="citation" className="py-16 px-6 bg-gray-50 border-y border-gray-100 scroll-mt-20">
      <div className="max-w-5xl mx-auto">
        <p className="text-sm uppercase tracking-widest text-brand font-medium mb-3">Code and citation</p>
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark tracking-tight">
          Inspect the experiments and cite the paper.
        </h2>
        <p className="text-gray-600 mt-4 max-w-3xl leading-relaxed">
          The repository includes the memory implementation, grounding and agent evaluations, PerceptMem harnesses, paper source, figures, and committed result files. Datasets, pretrained checkpoints, and cached encoder embeddings are excluded.
        </p>

        <div className="flex flex-wrap gap-3 mt-7">
          <a
            href={PAPER.repositoryUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-ink text-paper font-medium hover:bg-brand-dark transition-colors"
          >
            <Github size={16} /> GitHub repository
          </a>
          <a
            href={PAPER.abstractUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-gray-300 bg-white font-medium hover:bg-gray-100 transition-colors"
          >
            arXiv record <ExternalLink size={16} />
          </a>
        </div>

        <div className="mt-9 bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-bold mb-3">BibTeX</h3>
          <pre className="text-xs md:text-sm font-mono bg-gray-50 p-4 rounded overflow-x-auto leading-relaxed">
            {PAPER.bibtex}
          </pre>
        </div>
      </div>
    </section>
  );
}

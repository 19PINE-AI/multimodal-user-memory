import { ExternalLink, FileText } from 'lucide-react';
import { PAPER } from '../paper';

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-12 text-center">
        <p className="text-sm uppercase tracking-widest text-brand mb-4 font-medium">
          arXiv:{PAPER.arxivId} · cs.CL · 2026
        </p>
        <h1 className="font-serif font-bold text-5xl md:text-6xl tracking-tight leading-tight text-brand-dark">
          {PAPER.title}
        </h1>
        <p className="font-serif text-2xl md:text-3xl text-gray-600 mt-3 italic">
          {PAPER.subtitle}
        </p>

        <div className="mt-6 flex flex-wrap justify-center gap-x-8 gap-y-2 text-gray-700">
          {PAPER.authors.map((author) => (
            <span key={author.name}>
              <strong className="text-brand-dark">{author.name}</strong>
              <span className="text-gray-500"> · {author.affiliation}</span>
            </span>
          ))}
        </div>

        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <a
            href={PAPER.abstractUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-brand text-paper font-medium hover:bg-brand-dark transition-colors"
          >
            arXiv abstract <ExternalLink size={16} />
          </a>
          <a
            href={PAPER.pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-gray-300 font-medium hover:bg-gray-50 transition-colors"
          >
            Read PDF <FileText size={16} />
          </a>
        </div>
      </div>

      <div id="abstract" className="max-w-4xl mx-auto px-6 pb-14 scroll-mt-24">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-7 md:p-10 text-left">
          <h2 className="font-serif text-2xl font-bold text-brand-dark mb-5">Abstract</h2>
          {PAPER.abstract.map((paragraph) => (
            <p key={paragraph.slice(0, 40)} className="text-base md:text-lg leading-relaxed text-gray-800 mb-4 last:mb-0">
              {paragraph}
            </p>
          ))}
        </div>
      </div>
    </section>
  );
}

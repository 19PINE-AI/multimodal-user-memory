import { motion } from 'framer-motion';
import { ArrowRight, ExternalLink, FileText } from 'lucide-react';
import { PAPER } from '../paper';

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
        <motion.p
          initial={false}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-sm uppercase tracking-widest text-brand mb-4 font-medium"
        >
          arXiv:{PAPER.arxivId} · cs.CL · 2026
        </motion.p>
        <motion.h1
          initial={false}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-serif font-bold text-5xl md:text-6xl tracking-tight leading-tight text-brand-dark"
        >
          {PAPER.title}
        </motion.h1>
        <motion.p
          initial={false}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="font-serif text-2xl md:text-3xl text-gray-600 mt-3 italic"
        >
          {PAPER.subtitle}
        </motion.p>

        <motion.div
          initial={false}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-6 flex flex-wrap justify-center gap-x-8 gap-y-2 text-gray-700"
        >
          {PAPER.authors.map((author) => (
            <span key={author.name}>
              <strong className="text-brand-dark">{author.name}</strong>
              <span className="text-gray-500"> · {author.affiliation}</span>
            </span>
          ))}
        </motion.div>

        <motion.div
          initial={false}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-10 max-w-3xl mx-auto text-left"
        >
          <h2 className="font-serif text-2xl font-bold text-brand-dark mb-4">Abstract</h2>
          {PAPER.abstract.map((paragraph) => (
            <p key={paragraph.slice(0, 40)} className="mt-4 first:mt-0 text-base md:text-lg leading-relaxed text-gray-800">
              {paragraph}
            </p>
          ))}
        </motion.div>

        <motion.div
          initial={false}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-10 flex flex-wrap justify-center gap-3"
        >
          <a href="#key-results" className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-lg bg-brand text-paper font-medium hover:bg-brand-dark transition-colors">
            Headline results <ArrowRight size={16} />
          </a>
          <a href="#demo" className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-lg border border-gray-300 font-medium hover:bg-gray-50 transition-colors">
            Live mechanism demo
          </a>
          <a
            href={PAPER.abstractUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-lg border border-gray-300 font-medium hover:bg-gray-50 transition-colors"
          >
            arXiv abstract <ExternalLink size={16} />
          </a>
          <a
            href={PAPER.pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-lg border border-gray-300 font-medium hover:bg-gray-50 transition-colors"
          >
            Read PDF <FileText size={16} />
          </a>
        </motion.div>
      </div>

      {/* Architecture diagram banner */}
      <div className="max-w-6xl mx-auto px-6 pb-12">
        <div className="bg-white rounded-2xl shadow-md p-4 md:p-6 border border-gray-100">
          <img src={`${import.meta.env.BASE_URL}figs/fig0_arch.png`} alt="Architecture diagram" className="w-full" />
        </div>
        <p className="text-sm text-gray-500 mt-3 text-center italic">
          A frozen pretrained LM augmented by a per-modality bank of (encoder embedding, marker value embedding) pairs.
          The forward pre-hook on <code>lm_head</code> injects the cross-attention residual; ~8M trainable parameters over 3.1B frozen.
        </p>
      </div>
    </section>
  );
}

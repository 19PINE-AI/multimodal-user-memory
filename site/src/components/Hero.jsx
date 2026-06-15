import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      <div className="max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-sm uppercase tracking-widest text-brand mb-4 font-medium"
        >
          Companion to the paper
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-serif font-bold text-5xl md:text-6xl tracking-tight leading-tight text-brand-dark"
        >
          Parametric Multimodal User Memory
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="font-serif text-2xl md:text-3xl text-gray-600 mt-3 italic"
        >
          Storing what captions cannot carry
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-10 max-w-3xl mx-auto text-left"
        >
          <p className="text-base md:text-lg leading-relaxed text-gray-800">
            LM agents store user memory as <em>text</em> — transcripts, captions, summaries.
            This works for <strong>captionable</strong> content (<em>“my cat is named Bibi”</em>)
            but fails for <strong>perceptual</strong> content: how a user's voice sounds, what their
            face looks like across age, what their painter's brushwork looks like, how today's mood
            differs from baseline. Captioning destroys the discriminative signal.
          </p>
          <p className="mt-4 text-base md:text-lg leading-relaxed text-gray-800">
            We propose a <strong>parametric multimodal memory</strong>: a per-modality bank of
            (encoder embedding, value embedding) rows attached to a frozen pretrained LM via a
            forward pre-hook on <code className="font-mono text-sm bg-gray-100 px-1.5 py-0.5 rounded">lm_head</code>.
            Insertion is a single tensor append (<strong>O(1) wall-clock</strong>); query latency is flat
            at ~15 ms over <em>N</em> from 10 to 10,000.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
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

import { motion } from 'framer-motion';
import { CheckCircle2 } from 'lucide-react';

const designChoices = [
  {
    title: "No √D divisor",
    body: "With L2-normalised keys, dividing softmax logits by √D shrinks cosine-difference logits below 0.1 for D≥512, producing near-uniform attention. Drop the divisor.",
  },
  {
    title: "log β init = log 20",
    body: "Inverse-temperature init at 20 gives sharp attention at zero-shot. Init too low (β=1) → near-uniform softmax.",
  },
  {
    title: "Hook on lm_head pre-forward",
    body: "Attaching at lm_head pre-forward — not on an intermediate transformer layer — ensures the residual reaches logits without dilution.",
  },
  {
    title: "Learnable gain g = 8.0",
    body: "The natural logit boost from v·v ≈ ‖v‖² ≈ 1.2 is dwarfed by the LM's negative logit for unusual marker tokens. The scalar gain scales the residual to dominate.",
  },
];

export function Method() {
  return (
    <section id="method" className="py-16 px-6">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Method: continuous attention over a per-modality bank
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          The bank stores (key, value) pairs where the key is an L2-normalised encoder embedding
          and the value is the LM's embedding for an assigned marker token. At inference the LM's
          forward pre-hook on <code className="font-mono bg-gray-100 px-1.5 rounded">lm_head</code>
          adds a cross-attention residual that biases the next-token logit toward the matching marker.
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-6 md:p-8 mb-8">
          <p className="font-mono text-sm md:text-base text-gray-800 leading-loose">
            <span className="block">w = softmax(β · q·K<sup>⊤</sup>) ∈ ℝ<sup>N</sup></span>
            <span className="block">r = w<sup>⊤</sup>V ∈ ℝ<sup>H</sup></span>
            <span className="block">h' = h + g · W<sub>o</sub> r</span>
          </p>
          <p className="mt-4 text-sm text-gray-600">
            <strong>Trainable:</strong> W<sub>o</sub> (H×H, init I), scalar gain g (init 8.0),
            log inverse-temperature log β (init log 20), per-modality projection.
            Total ~8M trainable on top of 3.1B frozen.
          </p>
        </div>

        <h3 className="font-serif text-2xl font-bold mb-3 text-brand-dark">The four critical design choices</h3>
        <p className="text-gray-600 mb-4 text-sm">
          The architecture above is the <em>fourth</em> iteration. Three earlier attempts produced
          random-level retrieval (0.07 at N=5) despite plausible-looking loss curves. The four
          changes that turned random output into <strong>BEATS-RAG</strong>:
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          {designChoices.map((c, i) => (
            <motion.div
              key={c.title}
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.07 }}
              className="bg-white rounded-lg border border-gray-200 p-5"
            >
              <div className="flex items-start gap-2 mb-2">
                <CheckCircle2 className="text-accent-green flex-shrink-0 mt-0.5" size={18} />
                <h4 className="font-bold text-brand-dark">{c.title}</h4>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">{c.body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

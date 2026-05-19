import { motion } from 'framer-motion';
import { beatsRagRandom, beatsRagAdversarial } from '../data/results';
import { Trophy, Zap, FileLock, Globe } from 'lucide-react';

export function KeyResults() {
  return (
    <section id="key-results" className="py-16 px-6 bg-white border-y border-gray-100">
      <div className="max-w-6xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Seven multi-seed BEATS-RAG cells
        </h2>
        <p className="text-gray-600 mb-10 max-w-3xl">
          All cells verified at p&nbsp;&lt;&nbsp;0.05 (often p&nbsp;&lt;&nbsp;0.001). Δ is the absolute
          percentage-point gap over embedding-RAG cosine NN ceiling, the strongest text-free baseline.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {/* Random regime */}
          <div className="bg-gradient-to-br from-blue-50 to-white rounded-xl border border-blue-100 p-6">
            <div className="flex items-center gap-2 mb-3">
              <Trophy className="text-brand" size={20} />
              <h3 className="font-serif font-bold text-xl text-brand-dark">Random bank regime</h3>
            </div>
            <p className="text-sm text-gray-600 mb-4">Standard training; bank composed of randomly sampled identities.</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-gray-500 text-xs uppercase tracking-wide">
                  <th className="text-left py-2">Cell</th>
                  <th className="text-right py-2">n</th>
                  <th className="text-right py-2">RAG</th>
                  <th className="text-right py-2">AttMem</th>
                  <th className="text-right py-2 text-brand">Δ</th>
                  <th className="text-right py-2">p</th>
                </tr>
              </thead>
              <tbody>
                {beatsRagRandom.map((row, i) => (
                  <motion.tr
                    key={row.cell}
                    initial={{ opacity: 0, y: 6 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: i * 0.05 }}
                    className="border-b border-gray-100"
                  >
                    <td className="py-2 font-mono text-xs">{row.cell}</td>
                    <td className="text-right py-2 text-gray-500">{row.n}</td>
                    <td className="text-right py-2">{row.rag.toFixed(3)}</td>
                    <td className="text-right py-2 font-bold text-brand">{row.attmemMean.toFixed(3)} <span className="text-gray-400 font-normal">±{row.attmemStd.toFixed(3)}</span></td>
                    <td className="text-right py-2 font-bold text-accent-gold">+{(row.delta * 100).toFixed(1)}pp</td>
                    <td className="text-right py-2 text-gray-600 font-mono text-xs">{row.p.toFixed(3)}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Adversarial regime */}
          <div className="bg-gradient-to-br from-amber-50 to-white rounded-xl border border-amber-200 p-6">
            <div className="flex items-center gap-2 mb-3">
              <Trophy className="text-accent-gold" size={20} />
              <h3 className="font-serif font-bold text-xl text-brand-dark">Adversarial regime</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 font-medium">adv-training</span>
            </div>
            <p className="text-sm text-gray-600 mb-4">Bank composed of target + top-K most-cosine-similar distractors (K=19). Adv-training: 30% of training steps use adversarial banks.</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-gray-500 text-xs uppercase tracking-wide">
                  <th className="text-left py-2">Cell</th>
                  <th className="text-right py-2">n</th>
                  <th className="text-right py-2">RAG</th>
                  <th className="text-right py-2">AttMem-adv</th>
                  <th className="text-right py-2 text-accent-gold">Δ</th>
                </tr>
              </thead>
              <tbody>
                {beatsRagAdversarial.map((row, i) => (
                  <motion.tr
                    key={row.cell}
                    initial={{ opacity: 0, y: 6 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: i * 0.05 }}
                    className="border-b border-gray-100"
                  >
                    <td className="py-2 font-mono text-xs">{row.cell}</td>
                    <td className="text-right py-2 text-gray-500">{row.n}</td>
                    <td className="text-right py-2">{row.rag.toFixed(3)}</td>
                    <td className="text-right py-2 font-bold text-brand">
                      {row.attmemMean.toFixed(3)}
                      <span className="text-gray-400 font-normal"> ±{row.attmemStd.toFixed(3)}</span>
                    </td>
                    <td className="text-right py-2 font-bold text-accent-gold">+{(row.delta * 100).toFixed(1)}pp</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Quick-fact tiles */}
        <div className="grid md:grid-cols-4 gap-4">
          {[
            { icon: Zap, label: "Insertion", value: "0.5 ms / 1000 IDs", note: "O(1) wall-clock vs Path A's 1000 s" },
            { icon: Zap, label: "Query", value: "~15 ms", note: "flat over N from 10 to 10k" },
            { icon: Globe, label: "vs RAG-context", value: "52× faster", note: "RAG OOMs at N=10k" },
            { icon: FileLock, label: "Text non-regression", value: "byte-perfect", note: "top-1 8/8 on text prompts" },
          ].map(t => (
            <div key={t.label} className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-gray-500 mb-1">
                <t.icon size={14} />{t.label}
              </div>
              <div className="text-2xl font-bold text-brand-dark">{t.value}</div>
              <div className="text-xs text-gray-500 mt-1">{t.note}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

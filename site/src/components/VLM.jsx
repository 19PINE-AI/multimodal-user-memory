import { motion } from 'framer-motion';
import { vlmConfigs } from '../data/results';

export function VLM() {
  return (
    <section id="vlm" className="py-16 px-6">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Vision-language LM + key-value-space orthogonality
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          We test the mechanism on a real VLM (<strong>Qwen2.5-VL-3B-Instruct</strong>) processing raw face images.
          Two configurations differ only in the bank-key encoder — same frozen LM. The result:
          <strong> the bolt-on framework prefers an external, modality-specific encoder</strong>
          over the VLM's native vision tokens.
        </p>

        <div className="overflow-x-auto bg-white rounded-xl border border-gray-200 p-4 md:p-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-300 text-gray-700 text-xs uppercase tracking-wide">
                <th className="text-left py-3">Configuration</th>
                <th className="text-right py-3">N=10</th>
                <th className="text-right py-3">N=100</th>
                <th className="text-right py-3">N=1000</th>
              </tr>
            </thead>
            <tbody>
              {vlmConfigs.map((row, i) => (
                <motion.tr
                  key={i}
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className={`border-b border-gray-100 ${row.beats ? 'bg-amber-50' : ''}`}
                >
                  <td className="py-3 pr-4 text-sm">{row.config}</td>
                  <td className={`text-right py-3 font-mono ${row.beats ? 'font-bold text-accent-gold' : ''}`}>
                    {row.N10?.toFixed?.(3) ?? '—'}
                    {row.beats && <span className="ml-1 text-xs">BEATS</span>}
                  </td>
                  <td className="text-right py-3 font-mono">{row.N100?.toFixed?.(3) ?? '—'}</td>
                  <td className="text-right py-3 font-mono">{row.N1000?.toFixed?.(3) ?? '—'}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-6 bg-gradient-to-r from-blue-50 to-amber-50 rounded-xl border border-gray-200 p-6">
          <h3 className="font-serif font-bold text-lg text-brand-dark mb-2">
            The architectural finding
          </h3>
          <p className="text-sm text-gray-700 leading-relaxed">
            With <strong>ArcFace keys</strong> (512-d, orthogonal to LM's 2048-d hidden space), AttMem
            reproduces the Qwen2.5-3B BEATS-RAG result on Qwen-VL. With <strong>Qwen-VL's own vision
            tokens</strong> (already projected into LM hidden via the multimodal connector), AttMem
            <em> matches but cannot exceed</em> cosine NN. The BEATS comes from the LM's value-side
            prior adding signal in a <em>different</em> space than the key cosine; when key and value
            spaces coincide, the parametric memory degenerates back to encoder cosine NN.
          </p>
          <p className="mt-3 text-sm text-gray-700 leading-relaxed">
            <strong>Implication:</strong> the bolt-on framework prefers external, modality-specific
            encoders (ArcFace for faces, ECAPA for speakers, CLIP-mid for style) over the VLM's
            general-purpose vision encoder.
          </p>
        </div>
      </div>
    </section>
  );
}

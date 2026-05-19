import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

const advData = [
  { mode: 'A-XR-ID',  rag: 1.000, stdAttMem: 1.000, advAttMem: 1.000 },
  { mode: 'A-SCN',    rag: 0.827, stdAttMem: null,  advAttMem: 1.000 },
  { mode: 'A-PARA',   rag: 0.226, stdAttMem: 0.163, advAttMem: 0.934 },
  { mode: 'V-STY',    rag: 0.267, stdAttMem: 0.107, advAttMem: 0.977 },
  { mode: 'V-XC-ID',  rag: 0.841, stdAttMem: 0.808, advAttMem: 0.985 },
];

export function Adversarial() {
  return (
    <section id="adversarial" className="py-16 px-6 bg-amber-50/40 border-y border-amber-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Adversarial distractors: where standard training fails, adv-training transforms
        </h2>
        <p className="text-gray-600 mb-6 max-w-3xl">
          With adversarial banks (target + top-K most-cosine-similar non-matching identities),
          encoder cosine NN is genuinely confused. Standard-trained AttMem trails RAG by 2–16pp;
          mixing 30% adversarial banks into pretraining (<code>adv_prob = 0.3</code>) transforms
          the regime — multi-seed verified Δ over RAG of <strong className="text-accent-gold">+14pp / +17pp / +71pp / +71pp</strong>.
        </p>

        <div className="bg-white rounded-xl border border-amber-200 p-4 md:p-6 mb-6">
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={advData} margin={{ top: 30, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="mode" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1.1]} tick={{ fontSize: 12 }} label={{ value: 'retr@1 on adversarial K=19', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v?.toFixed?.(3) ?? 'n/a'} contentStyle={{ fontSize: 12 }} />
              <Bar dataKey="rag" name="RAG cosine NN" fill="#c44e52" />
              <Bar dataKey="stdAttMem" name="AttMem (standard training)" fill="#5a86b3" />
              <Bar dataKey="advAttMem" name="AttMem (adv-training)" fill="#e6a730" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Highlight A-PARA + V-STY +71pp wins */}
        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="bg-gradient-to-br from-amber-100 to-amber-50 border border-amber-300 rounded-xl p-6 text-center"
          >
            <div className="text-sm text-amber-800 uppercase tracking-wider mb-1">A-PARA paralinguistic state</div>
            <div className="text-5xl font-bold text-amber-700 my-2">+70.7<span className="text-2xl align-top">pp</span></div>
            <div className="text-sm text-gray-700">over RAG cosine on adversarial K=19, <span className="font-mono">n=4 seeds</span>, p&nbsp;&lt;&nbsp;0.001</div>
            <div className="text-xs text-gray-500 mt-2">RAG 0.226 → AttMem-adv 0.934 ± 0.004</div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-br from-amber-100 to-amber-50 border border-amber-300 rounded-xl p-6 text-center"
          >
            <div className="text-sm text-amber-800 uppercase tracking-wider mb-1">V-STY painter style</div>
            <div className="text-5xl font-bold text-amber-700 my-2">+71.0<span className="text-2xl align-top">pp</span></div>
            <div className="text-sm text-gray-700">over RAG cosine on adversarial K=19, <span className="font-mono">n=4 seeds</span>, p&nbsp;&lt;&nbsp;0.001</div>
            <div className="text-xs text-gray-500 mt-2">RAG 0.267 → AttMem-adv 0.977 ± 0.006</div>
          </motion.div>
        </div>

        <p className="text-sm text-gray-600 max-w-3xl italic">
          The largest wins come from sub-modalities where the encoder's cosine NN is genuinely weak
          on hard distractors (A-PARA, V-STY have weak encoders for cross-condition cases).
          Adv-training turns the encoder's adversarial weakness into the LM's strength.
        </p>
      </div>
    </section>
  );
}

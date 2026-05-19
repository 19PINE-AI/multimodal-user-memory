import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { crossFamily } from '../data/results';

export function CrossFamily() {
  return (
    <section id="cross-family" className="py-16 px-6 bg-white border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Cross-family generalisation
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          The architecture transfers across LM families. Llama-3.1-8B outperforms Qwen-3B at every
          large N. Mistral-7B-Instruct is recipe-sensitive (didn't converge at fixed 12K-step compute).
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6">
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={crossFamily} margin={{ top: 20, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="N" scale="log" domain={['dataMin', 'dataMax']} type="number"
                ticks={[5, 10, 20, 50, 100, 300, 700, 1000]}
                tick={{ fontSize: 12 }}
                label={{ value: 'N (registered identities)', position: 'insideBottom', offset: -2, style: { fontSize: 12 } }}
              />
              <YAxis domain={[0, 1.05]} tick={{ fontSize: 12 }} label={{ value: 'retr@1', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v?.toFixed?.(3)} contentStyle={{ fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="qwen3b"  name="Qwen2.5-3B (main)"     stroke="#1f4e79" strokeWidth={2.5} dot={{ r: 4 }} />
              <Line dataKey="qwen7b"  name="Qwen2.5-7B"            stroke="#e6a730" strokeWidth={2} dot={{ r: 3.5 }} />
              <Line dataKey="llama8b" name="Llama-3.1-8B"          stroke="#9c7cb5" strokeWidth={2.5} dot={{ r: 4 }} />
              <Line dataKey="mistral7b" name="Mistral-7B (recipe-sensitive)" stroke="#999" strokeWidth={1.5} strokeDasharray="3 3" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-6 grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="font-bold text-brand-dark mb-1">Llama-3.1-8B + adv-training</div>
            <p className="text-gray-700 leading-relaxed">
              Matches Qwen-3B+adv on adversarial K=19 (both 0.986, +14.5pp over RAG) but
              <em> preserves better random performance</em>: N=10 retr@1 0.90 vs Qwen-3B+adv's 0.83.
              Larger LM family acts as headroom that adv-training can exploit. <strong>Recommended config when both regimes matter.</strong>
            </p>
          </div>
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <div className="font-bold text-gray-700 mb-1">Mistral-7B: recipe-sensitive</div>
            <p className="text-gray-600 leading-relaxed">
              Default lr=3×10⁻⁴ + 12K steps: final loss 6.20 (vs Qwen 3.35). Needs smaller lr
              or longer training. Cross-family generalisation <em>holds but is not automatic at fixed compute</em>.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

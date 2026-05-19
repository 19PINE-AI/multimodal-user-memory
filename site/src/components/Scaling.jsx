import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, ReferenceDot } from 'recharts';
import { scalingCurve } from '../data/results';

export function Scaling() {
  return (
    <section id="scaling" className="py-16 px-6">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Scaling on V-XC-ID-XXXL (2180-ID face pool)
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          AttMem (trained, 12K-step curriculum) BEATS embedding-RAG at N=10 (p=0.006, n=4 seeds)
          and tracks within ~17pp through N=1000. Zero-shot AttMem falls off rapidly; the
          discrete-codebook predecessor (Path A) saturates at ~7%.
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={scalingCurve} margin={{ top: 20, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="N"
                scale="log"
                domain={['dataMin', 'dataMax']}
                type="number"
                tick={{ fontSize: 12 }}
                ticks={[5, 10, 20, 50, 100, 300, 700, 1000]}
                label={{ value: 'N (registered identities, log scale)', position: 'insideBottom', offset: -2, style: { fontSize: 12 } }}
              />
              <YAxis domain={[0, 1.05]} tick={{ fontSize: 12 }} label={{ value: 'retr@1', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v?.toFixed?.(3)} labelFormatter={(N) => `N = ${N}`} contentStyle={{ fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="rag"            name="RAG cosine NN (encoder ceiling)" stroke="#c44e52" strokeWidth={2} dot={{ r: 4 }} />
              <Line dataKey="attmem"         name="AttMem (trained)"                 stroke="#1f4e79" strokeWidth={2.5} dot={{ r: 4.5 }} />
              <Line dataKey="attmemZeroShot" name="AttMem (zero-shot)"               stroke="#5a86b3" strokeWidth={1.5} strokeDasharray="4 4" dot={{ r: 3 }} />
              <Line dataKey="pathA"          name="Path A (discrete codebook)"       stroke="#999999" strokeWidth={1.5} strokeDasharray="2 2" dot={{ r: 3 }} />
              <ReferenceDot x={10} y={0.992} r={8} fill="none" stroke="#e6a730" strokeWidth={2.5} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 text-center mt-3">
            Gold ring marks BEATS-RAG cell at N=10 (multi-seed p=0.006).
            AttMem retains 0.59 at N=1000 — vs Path A's ~0.07 = ~8× lift at scale.
          </p>
        </div>
      </div>
    </section>
  );
}

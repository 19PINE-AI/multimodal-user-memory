import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, ReferenceDot } from 'recharts';
import { latencyCurve } from '../data/results';
import { AlertCircle } from 'lucide-react';

// Filter out null data for chart rendering
const data = latencyCurve.map(d => ({ N: d.N, attmemQuery: d.attmemQuery, attmemInsert: d.attmemInsert, ragContext: d.ragContext }));

export function Latency() {
  return (
    <section id="latency" className="py-16 px-6 bg-white border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Latency: flat query, O(1) insertion
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          AttMem query latency is <strong>flat at ~15 ms</strong> regardless of N (bank matmul is
          microseconds; LM forward dominates). Batch insertion is ~0.5 ms total.
          RAG-with-LM-context grows linearly and OOMs at N=10,000 inside Qwen's 32k context window.
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6">
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={data} margin={{ top: 20, right: 24, left: 24, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="N" scale="log" domain={['dataMin', 'dataMax']} type="number"
                ticks={[10, 100, 1000, 10000]}
                tick={{ fontSize: 12 }}
                tickFormatter={(v) => v >= 1000 ? `${v/1000}k` : `${v}`}
                label={{ value: 'N (bank size, log scale)', position: 'insideBottom', offset: -2, style: { fontSize: 12 } }}
              />
              <YAxis
                scale="log" domain={[0.1, 'auto']} type="number"
                ticks={[0.1, 1, 10, 100, 1000]}
                tick={{ fontSize: 12 }}
                label={{ value: 'latency (ms, log scale)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
              />
              <Tooltip
                formatter={(v, name) => v == null ? 'OOM' : `${v.toFixed(2)} ms`}
                labelFormatter={(N) => `N = ${N}`}
                contentStyle={{ fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="attmemQuery"  name="AttMem query (LM forward)"  stroke="#1f4e79" strokeWidth={3} dot={{ r: 5 }} />
              <Line dataKey="attmemInsert" name="AttMem batch insert"        stroke="#5a86b3" strokeWidth={2}   dot={{ r: 4 }} />
              <Line dataKey="ragContext"   name="RAG-with-LM-context"        stroke="#c44e52" strokeWidth={2.5} dot={{ r: 4 }} connectNulls={false} />
              <ReferenceDot x={10000} y={50} r={6} fill="#c44e52" stroke="#fff" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 text-center mt-3 flex items-center justify-center gap-1.5">
            <AlertCircle size={14} className="text-rose-500" />
            RAG-with-LM-context OOMs at N=10,000 — exceeds Qwen's 32k-token context window. AttMem unaffected.
          </p>
        </div>

        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs uppercase tracking-wide text-gray-500">at N=1000</div>
            <div className="text-2xl font-bold text-brand-dark">52×</div>
            <div className="text-xs text-gray-500">faster than RAG-context</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs uppercase tracking-wide text-gray-500">at N=10k</div>
            <div className="text-2xl font-bold text-rose-600">∞</div>
            <div className="text-xs text-gray-500">RAG OOMs; AttMem 16.6 ms</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs uppercase tracking-wide text-gray-500">insertion</div>
            <div className="text-2xl font-bold text-brand-dark">~0.5 ms</div>
            <div className="text-xs text-gray-500">batch of any size</div>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
            <div className="text-xs uppercase tracking-wide text-gray-500">vs Path A</div>
            <div className="text-2xl font-bold text-accent-gold">2,000,000×</div>
            <div className="text-xs text-gray-500">at N=1000 insertion</div>
          </div>
        </div>
      </div>
    </section>
  );
}

import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';
import { perceptmemScorecard } from '../data/results';

const data = perceptmemScorecard.map(r => ({
  name: r.mode,
  sub: r.sub,
  'Path A (discrete codebook)': r.pathA,
  'RAG cosine NN': r.rag,
  'AttMem (ours)': r.attmem,
  beats: r.attmem > r.rag,
}));

export function Scorecard() {
  return (
    <section id="scorecard" className="py-16 px-6 bg-white border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          PerceptMem v0.2 scorecard
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          Cross-condition retrieval at N=10 across five sub-modalities. AttMem matches or exceeds the
          RAG cosine-NN encoder ceiling on four of five, and BEATS on V-XC-ID (multi-seed
          p=0.006) and V-STY (multi-seed p=0.009).
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6 mb-6">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={data} margin={{ top: 20, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" tick={{ fontSize: 12, fontFamily: 'Inter' }} />
              <YAxis domain={[0, 1.05]} tick={{ fontSize: 12 }} label={{ value: 'retr@1 at N=10', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
              <Tooltip formatter={(v) => v.toFixed(3)} contentStyle={{ fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="Path A (discrete codebook)" fill="#999999" />
              <Bar dataKey="RAG cosine NN" fill="#c44e52" />
              <Bar dataKey="AttMem (ours)" fill="#1f4e79">
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.beats ? '#e6a730' : '#1f4e79'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 text-center mt-3">
            ★ = AttMem BEATS RAG (V-XC-ID, V-STY at N=10; gold bars).
            A-XR-ID and A-SCN reach 0.90/0.83 — close to perfect-encoder ceilings of 1.00/0.93.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {perceptmemScorecard.map(r => (
            <div key={r.mode} className="bg-white border border-gray-200 rounded-lg p-3 text-sm">
              <div className="font-bold text-brand-dark">{r.mode}</div>
              <div className="text-xs text-gray-500 italic mb-2">{r.sub}</div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">RAG</span>
                <span>{r.rag.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">AttMem</span>
                <span className={r.attmem > r.rag ? 'font-bold text-accent-gold' : 'font-medium'}>{r.attmem.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Path A</span>
                <span className="text-gray-400">{r.pathA.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

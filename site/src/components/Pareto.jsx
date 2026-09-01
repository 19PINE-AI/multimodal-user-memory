import { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, LabelList } from 'recharts';
import { advProbPareto } from '../data/results';

const modalities = [
  { key: 'vxc', label: 'V-XC-ID-XXXL (2180 faces)', rndCol: 'vxcRandom', advCol: 'vxcAdv',     color: '#1f4e79' },
  { key: 'apara', label: 'A-PARA (paralinguistic)',  rndCol: 'aparaRandom', advCol: 'aparaAdv', color: '#5a86b3' },
  { key: 'vsty', label: 'V-STY (painter style)',     rndCol: 'vstyRandom', advCol: 'vstyAdv',   color: '#9c7cb5' },
];

export function Pareto() {
  const [active, setActive] = useState('vxc');
  const m = modalities.find(x => x.key === active);
  const points = advProbPareto.map(p => ({
    random: p[m.rndCol],
    adv: p[m.advCol],
    advProb: p.advProb,
  }));

  return (
    <section id="pareto" className="py-16 px-6">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          The <code className="font-mono text-3xl">adv_prob</code> Pareto frontier
        </h2>
        <p className="text-gray-600 mb-6 max-w-3xl">
          Random and adversarial performance trade off; the curve shape is itself modality-dependent.
          Pick a sub-modality to see the trade-off.
        </p>

        <div className="flex flex-wrap gap-2 mb-4">
          {modalities.map(x => (
            <button
              key={x.key}
              onClick={() => setActive(x.key)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                active === x.key ? 'bg-brand text-paper' : 'bg-white border border-gray-300 hover:bg-gray-50'
              }`}
              style={active === x.key ? {} : { color: x.color }}
            >
              {x.label}
            </button>
          ))}
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6 mb-4">
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart margin={{ top: 30, right: 30, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="random" type="number" domain={[0, 1.05]} tick={{ fontSize: 12 }} label={{ value: 'random N=10 retr@1', position: 'insideBottom', offset: -8, style: { fontSize: 12 } }} />
              <YAxis dataKey="adv" type="number" domain={[0, 1.05]} tick={{ fontSize: 12 }} label={{ value: 'adversarial K=19 retr@1', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
              <Tooltip
                formatter={(v) => v.toFixed(3)}
                labelFormatter={() => ''}
                content={({ payload }) => {
                  if (!payload || !payload.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded shadow-md p-2 text-xs">
                      <div className="font-mono font-bold">adv_prob = {d.advProb}</div>
                      <div>random N=10: {d.random.toFixed(3)}</div>
                      <div>adv K=19: {d.adv.toFixed(3)}</div>
                    </div>
                  );
                }}
              />
              <Scatter data={points} fill={m.color} line={{ stroke: m.color, strokeWidth: 2 }} shape="circle">
                <LabelList dataKey="advProb" position="right" formatter={(v) => `p=${v}`} style={{ fontSize: 11, fill: '#444' }} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="grid md:grid-cols-3 gap-4 text-sm">
          {modalities.map(x => {
            return (
              <div key={x.key} className={`rounded-lg border p-4 ${active === x.key ? 'border-brand bg-blue-50' : 'border-gray-200 bg-white'}`}>
                <div className="font-bold" style={{ color: x.color }}>{x.label}</div>
                {x.key === 'vxc' && (
                  <p className="text-xs text-gray-600 mt-2">Gradual Pareto. Sweet spot <code className="font-mono">adv_prob=0.1</code> retains 0.87 random while reaching 0.98 adversarial.</p>
                )}
                {x.key === 'apara' && (
                  <p className="text-xs text-gray-600 mt-2">Sharp drop. Even <code className="font-mono">adv_prob=0.1</code> cuts random from 0.47 → 0.30, but adversarial jumps from 0.16 → 0.91 (+75pp).</p>
                )}
                {x.key === 'vsty' && (
                  <p className="text-xs text-gray-600 mt-2">Default <code className="font-mono">adv_prob=0</code> already BEATS RAG on random (0.47 vs 0.40). Use only for known-adversarial deployments.</p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

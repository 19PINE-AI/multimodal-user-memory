import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, ReferenceLine, ReferenceArea } from 'recharts';
import { trainingMatters } from '../data/results';

// Build combined dataset by N
const allNs = [...new Set(Object.values(trainingMatters).flatMap(arr => arr.map(d => d.N)))].sort((a, b) => a - b);
const combined = allNs.map(N => {
  const row = { N };
  for (const [mode, data] of Object.entries(trainingMatters)) {
    const point = data.find(d => d.N === N);
    if (point) row[mode] = point.delta;
  }
  return row;
});

export function TrainingMatters() {
  return (
    <section id="training-matters" className="py-16 px-6 bg-white">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-3 tracking-tight">
          Training matters: when the parametric memory adds value over cosine
        </h2>
        <p className="text-gray-700 max-w-3xl leading-relaxed mb-6">
          To disambiguate "what does the parametric memory add beyond cosine?", we evaluate
          at <code className="font-mono text-sm bg-gray-100 px-1.5 py-0.5 rounded">n_steps=0</code> (every parameter
          at initialisation: W<sub>o</sub>=I, g=8, log β=log 20) and compare to the trained model.
          Three regimes emerge.
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6 mb-6">
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={combined} margin={{ top: 16, right: 24, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="N" scale="log" domain={['dataMin', 'dataMax']} type="number"
                     ticks={[5, 10, 20, 50, 100, 300, 700]} tick={{ fontSize: 12 }}
                     label={{ value: 'N (registered identities)', position: 'insideBottom', offset: -2, style: { fontSize: 12 } }} />
              <YAxis domain={[-0.2, 0.7]} tick={{ fontSize: 12 }}
                     label={{ value: 'Δ retr@1 (trained − zero-shot)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
              <ReferenceArea y1={-0.2} y2={0} fill="#fee2e2" fillOpacity={0.4} />
              <ReferenceArea y1={0} y2={0.7} fill="#dcfce7" fillOpacity={0.3} />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
              <Tooltip contentStyle={{ fontSize: 12 }} formatter={(v) => v?.toFixed?.(3)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="A-PARA"  stroke="#5a86b3" strokeWidth={2} dot={{ r: 4 }} connectNulls />
              <Line dataKey="A-XR-ID" stroke="#c44e52" strokeWidth={2} dot={{ r: 4 }} connectNulls />
              <Line dataKey="A-SCN"   stroke="#3a8c5d" strokeWidth={2} dot={{ r: 4 }} connectNulls />
              <Line dataKey="V-STY"   stroke="#9c7cb5" strokeWidth={2} dot={{ r: 4 }} connectNulls />
              <Line dataKey="V-XC-ID" stroke="#1f4e79" strokeWidth={2.5} dot={{ r: 4.5 }} connectNulls />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Green region: training helps. Red region: training hurts (A-XR-ID, encoder ceiling 1.00).
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="rounded-lg border border-green-200 bg-green-50/50 p-4">
            <div className="font-bold text-green-800 mb-1">① Training drives the win at scale</div>
            <p className="text-gray-700 leading-relaxed">
              On V-XC-ID-XXXL, zero-shot is below RAG at every N; trained beats RAG at N=10.
              Lift grows from <strong>+0.13 at N=10</strong> to <strong>+0.40 at N=700</strong>.
              Pretraining produces real per-modality discriminative power.
            </p>
          </div>
          <div className="rounded-lg border border-rose-200 bg-rose-50/50 p-4">
            <div className="font-bold text-rose-800 mb-1">② Training hurts when encoder is perfect</div>
            <p className="text-gray-700 leading-relaxed">
              On A-XR-ID (RAG=1.00), zero-shot matches the ceiling exactly; trained W<sub>o</sub>
              adds noise that costs ~0.10. <strong>Honest result:</strong> when cosine is at 1.00,
              the LM has nothing useful to add.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50/50 p-4">
            <div className="font-bold text-amber-800 mb-1">③ kNN-LM alone can beat cosine</div>
            <p className="text-gray-700 leading-relaxed">
              On V-STY N=5, zero-shot reaches 0.53 vs RAG 0.40. The bank's structural mechanism
              (soft histogram over value-side embeddings) suffices at small N on low-ceiling
              encoders.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';

const pivotData = [
  { mode: 'A-XR-ID', N: '10', pathA: 0.32, attmem: 0.90 },
  { mode: 'A-SCN',   N: '10', pathA: 0.40, attmem: 0.83 },
  { mode: 'A-PARA',  N: '10', pathA: 0.45, attmem: 0.44 },
  { mode: 'V-STY',   N: '5',  pathA: 0.20, attmem: 0.64 },
  { mode: 'V-XC-ID', N: '10', pathA: 0.10, attmem: 0.99 },
  { mode: 'V-XC-ID', N: '700', pathA: 0.07, attmem: 0.63 },
];

const kSweepData = [
  { K: 32,   sameCode: 0.32, gateRet: 0.51, netRetr1: 0.07 },
  { K: 64,   sameCode: 0.38, gateRet: 0.44, netRetr1: 0.07 },
  { K: 128,  sameCode: 0.43, gateRet: 0.36, netRetr1: 0.07 },
  { K: 256,  sameCode: 0.46, gateRet: 0.29, netRetr1: 0.07 },
  { K: 512,  sameCode: 0.50, gateRet: 0.22, netRetr1: 0.07 },
  { K: 1024, sameCode: 0.53, gateRet: 0.16, netRetr1: 0.07 },
];

export function PathA() {
  return (
    <section id="path-a" className="py-16 px-6 bg-gray-50 border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-3 tracking-tight">
          Why captioning intermediates fail: the design-space finding
        </h2>
        <p className="text-gray-700 max-w-3xl mb-4 leading-relaxed">
          We tuned <strong>Path A</strong> (a discrete-codebook predecessor) over 16 development
          cycles before pivoting. The codebook is itself a learned analog of text-captioning's
          information loss: any intermediate categorical representation discards the perceptual
          signal the encoder preserved.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Path A K-sweep collapse */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-bold text-brand-dark mb-2 text-sm uppercase tracking-wide">Path A K-sweep on V-XC-ID-XXXL</h3>
            <p className="text-xs text-gray-600 mb-3">
              Increasing codebook size K from 32 to 1024 lifts same-code rate (0.32 → 0.53) but
              gate retrieval collapses (0.51 → 0.16) — net retr@1 unchanged at ~0.07.
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={kSweepData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="K" tick={{ fontSize: 11 }} label={{ value: 'codebook K', position: 'insideBottom', offset: -2, style: { fontSize: 11 } }} />
                <YAxis domain={[0, 0.6]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="sameCode" name="codebook same-code rate" fill="#5a86b3" />
                <Bar dataKey="gateRet"  name="gate retrieval"          fill="#c44e52" />
                <Bar dataKey="netRetr1" name="net retr@1"               fill="#999999" />
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-500 italic mt-2">
              Encoder upgrade (R50 → AntelopeV2 R100 on 360K IDs) and 100K-step continual pretraining
              do not move the needle either. The quantisation step is the binding constraint.
            </p>
          </div>

          {/* Path A → AttMem per modality */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-bold text-brand-dark mb-2 text-sm uppercase tracking-wide">Pivot per sub-modality</h3>
            <p className="text-xs text-gray-600 mb-3">
              Replacing the discrete codebook with continuous attention yields 2–10× retr@1 across
              sub-modalities at N=10 (V-XC-ID also at N=700).
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={pivotData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="mode" tick={{ fontSize: 10 }}
                  tickFormatter={(v, idx) => `${v}\nN=${pivotData[idx]?.N}`}
                  interval={0}
                  height={36}
                />
                <YAxis domain={[0, 1.05]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="pathA"  name="Path A (codebook)"           fill="#999999" />
                <Bar dataKey="attmem" name="AttMem (continuous)" fill="#1f4e79" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border-2 border-brand p-5">
          <p className="text-sm text-gray-800 leading-relaxed">
            <strong className="text-brand-dark">The quantisation step itself is the binding constraint.</strong>
            Two encoder embeddings that fall in the same codebook cell are indistinguishable downstream;
            at N≥300 the collision rate dominates retr@1. Continuous attention over the raw embedding
            sidesteps the quantisation entirely — and pays only an <code className="font-mono text-xs bg-gray-100 px-1 rounded">O(N·D)</code>
            matmul per query (microseconds at any reasonable N).
          </p>
        </div>
      </div>
    </section>
  );
}

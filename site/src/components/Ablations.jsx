import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, ReferenceDot } from 'recharts';

const lmSize = [
  { N: 5,    r: 0.93, q3b_12: 0.93, q3b_50: 0.93, q7b_12: 1.00, q7b_50: 1.00 },
  { N: 10,   r: 0.93, q3b_12: 0.99, q3b_50: 0.97, q7b_12: 1.00, q7b_50: 1.00 },
  { N: 20,   r: 0.80, q3b_12: 0.81, q3b_50: 0.82, q7b_12: 0.80, q7b_50: 0.83 },
  { N: 50,   r: 0.77, q3b_12: 0.73, q3b_50: 0.75, q7b_12: 0.72, q7b_50: 0.73 },
  { N: 100,  r: 0.78, q3b_12: 0.74, q3b_50: 0.76, q7b_12: 0.74, q7b_50: 0.74 },
  { N: 300,  r: 0.73, q3b_12: 0.64, q3b_50: 0.66, q7b_12: 0.62, q7b_50: 0.64 },
  { N: 700,  r: 0.76, q3b_12: 0.63, q3b_50: 0.65, q7b_12: 0.58, q7b_50: 0.62 },
  { N: 1000, r: 0.77, q3b_12: 0.59, q3b_50: 0.63, q7b_12: 0.50, q7b_50: 0.57 },
];

const curriculum = [
  { N: 5,    rag: 0.93, fixed: 0.93, curriculum: 0.93 },
  { N: 10,   rag: 0.93, fixed: 1.00, curriculum: 0.99 },
  { N: 20,   rag: 0.80, fixed: 0.83, curriculum: 0.81 },
  { N: 50,   rag: 0.77, fixed: 0.73, curriculum: 0.73 },
  { N: 100,  rag: 0.78, fixed: 0.56, curriculum: 0.74 },
  { N: 300,  rag: 0.73, fixed: 0.29, curriculum: 0.64 },
  { N: 700,  rag: 0.76, fixed: 0.20, curriculum: 0.63 },
];

export function Ablations() {
  return (
    <section id="ablations" className="py-16 px-6 bg-gray-50 border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-3 tracking-tight">
          Ablations: LM size × training steps, curriculum bank size
        </h2>
        <p className="text-gray-700 max-w-3xl leading-relaxed mb-6">
          Two ablations on V-XC-ID-XXXL clarify the recipe: <strong>(a)</strong> Larger LM is not
          automatically better within fixed compute; <strong>(b)</strong> Curriculum bank size is
          essential at large N.
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="font-bold text-sm uppercase tracking-wider text-gray-500 mb-2">(a) LM size × steps</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lmSize} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="N" scale="log" type="number" domain={['dataMin', 'dataMax']}
                       ticks={[5, 10, 50, 100, 300, 1000]} tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1.05]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="r"      name="RAG ceiling"     stroke="#000"     strokeWidth={1.2} strokeOpacity={0.5} dot={false} />
                <Line dataKey="q3b_12" name="3B @ 12K"  stroke="#1f4e79" strokeWidth={2} dot={{ r: 3 }} />
                <Line dataKey="q3b_50" name="3B @ 50K"  stroke="#1f4e79" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} />
                <Line dataKey="q7b_12" name="7B @ 12K"  stroke="#e69f00" strokeWidth={2} dot={{ r: 3 }} />
                <Line dataKey="q7b_50" name="7B @ 50K"  stroke="#e69f00" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-600 mt-2">
              Qwen2.5-7B (untied embeddings, our <code>lm_head.weight</code> value fix) beats 3B at
              N≤20 but lags at scale within 12K steps. At 50K steps the gap narrows but doesn't close
              (3B@50K @ N=1000: 0.625; 7B@50K: 0.569).
            </p>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="font-bold text-sm uppercase tracking-wider text-gray-500 mb-2">(b) Curriculum bank size</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={curriculum} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="N" scale="log" type="number" domain={['dataMin', 'dataMax']}
                       ticks={[5, 10, 50, 100, 300, 700]} tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1.05]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="rag" name="RAG ceiling" stroke="#000" strokeWidth={1.2} strokeOpacity={0.5} dot={false} />
                <Line dataKey="fixed"      name="bs=64 fixed (8K)"    stroke="#c44e52" strokeWidth={2} dot={{ r: 3 }} />
                <Line dataKey="curriculum" name="bs∈[64,1024] (12K)"   stroke="#1f4e79" strokeWidth={2.5} dot={{ r: 3.5 }} />
                <ReferenceDot x={700} y={0.63} r={6} fill="none" stroke="#e6a730" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-600 mt-2">
              Fixed <code>bs=64</code> causes a train/eval distribution shift at large N (retr@1
              drops from 0.63 to 0.20 at N=700). Curriculum bank-size sampling
              <code>bs ∼ Uniform[64, 1024]</code> closes the gap by <strong className="text-accent-gold">+0.43 retr@1 at N=700</strong>.
            </p>
          </div>
        </div>

        <div className="mt-6 grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <strong>Tied vs untied embeddings.</strong> Auto-detected. For Qwen2.5-3B (tied),
            <code>v = input_embedding[marker]</code>; for Qwen2.5-7B and Llama-3.1-8B (untied),
            <code>v = lm_head.weight[marker]</code>. Without the fix, the residual addition
            produces a cross-product of unrelated vectors instead of <code>‖v‖²</code>.
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <strong>Curriculum is essential for scaling.</strong> Without it, AttMem at N=700 collapses
            to 0.20 (below Path A). With <code>bs ∼ Uniform[64,1024]</code>, AttMem reaches 0.63 —
            the gap between in-distribution (N≈64) and out-of-distribution (N=700) eval is closed.
          </div>
        </div>
      </div>
    </section>
  );
}

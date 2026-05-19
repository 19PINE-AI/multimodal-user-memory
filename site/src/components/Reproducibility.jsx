import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

const snippets = {
  'Random-regime BEATS-RAG (V-XC-ID, p=0.006, n=4)': `for s in 42 43 44 47; do
  python3 src/nanochat_mm/attmem_train_and_eval.py \\
      v-xc-id-xxxl 12000 $s 1024
done`,
  'Adversarial-regime BEATS-RAG (multi-seed, p<0.001)': `for s in 49 50 51; do
  python3 src/nanochat_mm/attmem_train_and_eval.py \\
      v-xc-id-xxxl 12000 $s 1024 0.3
done

for s in 42 43 44 45; do
  python3 src/nanochat_mm/attmem_train_and_eval.py a-para     5000 $s 0   0.3
  python3 src/nanochat_mm/attmem_train_and_eval.py v-sty-clip 5000 $s 0   0.3
  python3 src/nanochat_mm/attmem_train_and_eval.py a-scn      5000 $s 0   0.3
done`,
  'Cross-family (Llama-3.1-8B)': `ATTMEM_MODEL_ID="NousResearch/Meta-Llama-3.1-8B-Instruct" \\
  python3 src/nanochat_mm/attmem_train_and_eval.py \\
      v-xc-id-xxxl 12000 42 1024 0.3`,
  'VLM end-to-end (Qwen2.5-VL)': `# Zero-shot AttMem on Qwen-VL with native vision tokens
python3 src/nanochat_mm/attmem_vl_eval.py 10 42

# Pretrained with cached visual keys
python3 src/nanochat_mm/attmem_vl_train.py 3000 42 0

# VL + external ArcFace keys (validates key-value orthogonality)
python3 src/nanochat_mm/attmem_vl_arcface.py 12000 42 1024`,
  'Latency + propositional control': `python3 src/nanochat_mm/attmem_latency_benchmark.py
python3 src/nanochat_mm/attmem_propositional_control.py`,
};

export function Reproducibility() {
  return (
    <section id="reproducibility" className="py-16 px-6 bg-gray-50 border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Reproducibility
        </h2>
        <p className="text-gray-600 mb-6 max-w-3xl">
          The full system is ~200 lines of new code plus the frozen-LM <code>transformers</code> stack.
          CLI signature: <code>mode, n_steps, seed, [bank_size_max], [adv_prob]</code>. Each run takes
          ~5–17 min on an H100-class GPU and logs to <code>results/</code>.
        </p>

        <div className="space-y-6">
          {Object.entries(snippets).map(([title, code]) => (
            <div key={title}>
              <h3 className="font-bold text-sm text-brand-dark mb-2">{title}</h3>
              <div className="rounded-lg overflow-hidden border border-gray-200 bg-white">
                <SyntaxHighlighter
                  language="bash"
                  style={oneLight}
                  customStyle={{ margin: 0, padding: '1rem', fontSize: '13px', background: '#fff' }}
                  wrapLongLines
                >
                  {code}
                </SyntaxHighlighter>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-bold mb-2">BibTeX</h3>
          <pre className="text-xs font-mono bg-gray-50 p-3 rounded overflow-x-auto">
{`@misc{li2026pmum,
  title  = {Parametric Multimodal User Memory: Storing What Captions Cannot Carry},
  author = {Li, Bojie},
  year   = {2026},
  note   = {\\url{https://github.com/bojieli/multimodal-user-memory}}
}`}
          </pre>
        </div>
      </div>
    </section>
  );
}

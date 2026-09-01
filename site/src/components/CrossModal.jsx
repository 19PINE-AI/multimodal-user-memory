import { CheckCircle2, Eye, Mic, FileText } from 'lucide-react';

export function CrossModal() {
  return (
    <section id="cross-modal" className="py-16 px-6 bg-white">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-3 tracking-tight">
          Cross-modal independence + text-recall non-regression
        </h2>
        <p className="text-gray-700 max-w-3xl leading-relaxed mb-6">
          The parametric memory composes <em>additively</em> with existing text memory. Two
          structural validations confirm the composition is clean.
        </p>

        <div className="grid md:grid-cols-2 gap-5">
          {/* Cross-modal */}
          <div className="rounded-xl border border-gray-200 bg-white p-6">
            <div className="flex items-center gap-2 mb-3">
              <Eye className="text-brand" size={20} />
              <Mic className="text-accent-rose" size={20} />
              <h3 className="font-bold text-brand-dark">Cross-modal independence</h3>
            </div>
            <p className="text-sm text-gray-700 mb-4 leading-relaxed">
              We register 20 face IDs and 15 speaker IDs in the <em>same</em> model and let argmax
              span the union of all markers (zero-shot; no per-modality training).
            </p>
            <div className="bg-gray-50 rounded-lg p-3 text-sm space-y-2 font-mono">
              <div className="flex justify-between">
                <span className="text-gray-700">Vision query retr@1:</span>
                <span className="font-bold">0.77</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>cross-modal leak (vision → audio markers):</span>
                <span>0.017</span>
              </div>
              <hr className="border-gray-200" />
              <div className="flex justify-between">
                <span className="text-gray-700">Audio query retr@1:</span>
                <span className="font-bold">0.93</span>
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>cross-modal leak (audio → vision markers):</span>
                <span>0.067</span>
              </div>
            </div>
            <p className="text-xs text-gray-600 italic mt-3">
              The per-modality banks are independent <em>by construction</em> — without any
              per-modality training, leak is already &lt;7%.
            </p>
          </div>

          {/* Text non-regression */}
          <div className="rounded-xl border border-gray-200 bg-white p-6">
            <div className="flex items-center gap-2 mb-3">
              <FileText className="text-accent-green" size={20} />
              <h3 className="font-bold text-brand-dark">Text-recall non-regression</h3>
            </div>
            <p className="text-sm text-gray-700 mb-4 leading-relaxed">
              On 8 propositional English prompts ("The capital of France is", etc.), top-1 next-token
              prediction is byte-identical to vanilla Qwen across all configurations.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2 p-2 bg-green-50 border border-green-200 rounded">
                <CheckCircle2 className="text-accent-green" size={16} />
                <span className="flex-1">Hook installed but no-op</span>
                <span className="font-mono text-xs">byte-perfect</span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-green-50 border border-green-200 rounded">
                <CheckCircle2 className="text-accent-green" size={16} />
                <span className="flex-1">Bolt forward, empty bank</span>
                <span className="font-mono text-xs">top-1 8/8</span>
              </div>
              <div className="flex items-center gap-2 p-2 bg-green-50 border border-green-200 rounded">
                <CheckCircle2 className="text-accent-green" size={16} />
                <span className="flex-1">Bolt forward, populated banks</span>
                <span className="font-mono text-xs">top-1 8/8</span>
              </div>
            </div>
            <p className="text-xs text-gray-600 italic mt-3">
              The hook itself is byte-perfect on text inputs. Propositional and perceptual memory
              primitives compose cleanly — no regression on the LM's normal text-generation behaviour.
            </p>
          </div>
        </div>

        <div className="mt-6 rounded-xl border-2 border-brand bg-blue-50 p-5 text-sm text-brand-dark leading-relaxed">
          <strong>An agent with both primitives strictly dominates one with only text memory.</strong>
          Captionable content ("my favourite restaurant is in Paris") goes to the text store and is
          reliably retrieved by sentence-encoder cosine. Perceptual content ("how Bibi looks across
          lighting and age") goes to the parametric memory. The two compose; neither replaces the other.
        </div>
      </div>
    </section>
  );
}

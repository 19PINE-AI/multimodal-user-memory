import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { namedDemo } from '../data/results';
import { Play, Pause, RotateCcw, CheckCircle2 } from 'lucide-react';

export function Demo() {
  const [step, setStep] = useState(0);  // 0 = registration phase, 1..10 = queries
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= namedDemo.length) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 1500);
    return () => clearInterval(id);
  }, [playing]);

  const phase = step === 0 ? 'register' : step > namedDemo.length ? 'done' : 'query';
  const currentQuery = step > 0 && step <= namedDemo.length ? namedDemo[step - 1] : null;
  const correctCount = step > namedDemo.length
    ? namedDemo.filter(q => q.correct).length
    : namedDemo.slice(0, Math.max(step - 1, 0)).filter(q => q.correct).length;

  const reset = () => { setStep(0); setPlaying(false); };

  return (
    <section id="demo" className="py-16 px-6 bg-white border-y border-gray-100">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          Live mechanism demo: register-then-recall
        </h2>
        <p className="text-gray-600 mb-6 max-w-3xl">
          Register 10 AgeDB celebrities by their real first names as single-token markers (Tom,
          Sean, Grace, etc.), then query with held-out images. Zero-shot AttMem (no pretraining
          of bolt parameters) hits <strong>10/10 retr@1</strong>.
        </p>

        <div className="flex items-center gap-2 mb-4">
          <button
            onClick={() => setPlaying(!playing)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-brand text-paper text-sm font-medium hover:bg-brand-dark transition-colors"
          >
            {playing ? <Pause size={14} /> : <Play size={14} />}
            {playing ? 'Pause' : step === 0 ? 'Start demo' : step > namedDemo.length ? 'Replay' : 'Resume'}
          </button>
          <button
            onClick={reset}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-gray-300 text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            <RotateCcw size={14} /> Reset
          </button>
          <span className="text-sm text-gray-500 ml-2">
            Step {step} / {namedDemo.length} · {correctCount}/{Math.max(step - (phase === 'register' ? 0 : 0), 0)} correct
          </span>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Bank visualization */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-bold text-sm uppercase tracking-wide text-gray-500 mb-3">Bank (register phase)</h3>
            <div className="grid grid-cols-2 gap-2">
              {namedDemo.map((entry, i) => (
                <motion.div
                  key={entry.name}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{
                    opacity: 1, x: 0,
                    backgroundColor: currentQuery && entry.name === currentQuery.name ? '#fef3c7' : 'transparent'
                  }}
                  transition={{ duration: 0.3, delay: phase === 'register' ? i * 0.05 : 0 }}
                  className="border border-gray-200 rounded-md px-3 py-2 text-sm flex items-center gap-2"
                >
                  <span className="font-mono text-xs text-gray-400">#{i+1}</span>
                  <span className="font-medium">{entry.name}</span>
                  {currentQuery && entry.name === currentQuery.name && (
                    <span className="ml-auto text-xs text-amber-700 font-bold">target</span>
                  )}
                </motion.div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-3 italic">
              Each row = (ArcFace 512-d face embedding, LM input embedding of first-name token).
              Bank populated by 10 sequential <code className="text-xs">torch.cat</code> calls (~0.5 ms).
            </p>
          </div>

          {/* Query visualization */}
          <div className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-bold text-sm uppercase tracking-wide text-gray-500 mb-3">Query (held-out face)</h3>
            <AnimatePresence mode="wait">
              {phase === 'register' && (
                <motion.div key="register" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="text-center py-12 text-gray-400">
                  Press <strong>Start demo</strong> to begin querying.
                </motion.div>
              )}
              {phase === 'query' && currentQuery && (
                <motion.div key={`q-${step}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                  className="space-y-3">
                  <div className="text-sm text-gray-600">
                    Querying with a different photo of <strong className="text-brand-dark">{currentQuery.name}</strong>...
                  </div>
                  <div className="bg-gray-50 rounded p-3 font-mono text-xs space-y-1">
                    <div className="text-gray-500">// LM forward via lm_head pre-hook (bank residual injected)</div>
                    <div>logits[<span className="text-brand">"You see"</span>, ..., <span className="text-accent-gold">[face_emb]</span>] →</div>
                    <div className="text-gray-500">// top-3 marker-restricted argmax:</div>
                    {currentQuery.top3.map(([name, logit], i) => (
                      <div key={i} className={i === 0 ? 'text-brand-dark font-bold' : 'text-gray-600'}>
                        {i + 1}. {name} ({logit.toFixed(1)}){i === 0 && currentQuery.correct && <span className="text-accent-green ml-2">✓ correct</span>}
                      </div>
                    ))}
                  </div>
                  <div className={`rounded-md p-3 text-sm font-medium flex items-center gap-2 ${
                    currentQuery.correct ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-rose-50 text-rose-800 border border-rose-200'
                  }`}>
                    {currentQuery.correct ? <CheckCircle2 size={16} /> : null}
                    Predicted: <strong>{currentQuery.pred}</strong> (target: {currentQuery.name})
                  </div>
                </motion.div>
              )}
              {phase === 'done' && (
                <motion.div key="done" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-8">
                  <CheckCircle2 className="text-accent-green mx-auto mb-3" size={48} />
                  <div className="text-3xl font-bold text-brand-dark">10 / 10</div>
                  <div className="text-sm text-gray-600 mt-2">
                    retr@1 — zero-shot (no bolt pretraining), restricted argmax over 10 registered markers.
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <p className="mt-6 text-sm text-gray-600 max-w-3xl">
          The mechanism's strength here: with the bank populated, the LM's <code>lm_head</code>
          pre-hook adds a logit boost on the matching marker. Since markers are real
          single-token first names (Tom = 24732, Sean = 59816, etc.), the LM's natural decoding
          continues to be valid text. This is what "the parametric memory composes with text decoding" means concretely.
        </p>
      </div>
    </section>
  );
}

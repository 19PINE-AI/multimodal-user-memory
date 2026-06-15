export function Mechanism() {
  return (
    <section id="mechanism" className="py-16 px-6">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-2 tracking-tight">
          What the mechanism is doing
        </h2>
        <p className="text-gray-600 mb-8 max-w-3xl">
          Three panels showing the chain encoder → bank attention → LM amplification on a held-out
          face cross-condition probe (N=10). Diagonal-dominant attention with LM-side adjustments.
        </p>

        <div className="bg-white rounded-xl border border-gray-200 p-4 md:p-6">
          <img src={`${import.meta.env.BASE_URL}figs/fig9_mechanism.png`} alt="Mechanism analysis" className="w-full" />
          <p className="text-sm text-gray-600 mt-4">
            <strong>(a) Bank attention weights:</strong> sharp diagonal — queries attend overwhelmingly
            to their correct target identity. <strong>(b) Encoder cosine similarity:</strong> diagonal-dominant
            but with off-diagonal noise. <strong>(c) LM marker logits (row-normalised):</strong> inherit the
            attention diagonal + LM-side adjustments that further sharpen the argmax.
          </p>
        </div>

        <div className="mt-6 bg-gradient-to-r from-blue-50 to-amber-50 rounded-xl border border-gray-200 p-6">
          <h3 className="font-serif font-bold text-lg text-brand-dark mb-2">
            Three regimes when AttMem adds value
          </h3>
          <div className="space-y-3 text-sm text-gray-700">
            <div className="flex gap-3 items-start">
              <span className="text-accent-gold font-bold mt-0.5">①</span>
              <div>
                <strong>Random bank + imperfect encoder:</strong> AttMem standard training BEATS RAG
                by extracting LM-side signal orthogonal to encoder cosine (V-STY +24pp, V-XC-ID +5.9pp).
              </div>
            </div>
            <div className="flex gap-3 items-start">
              <span className="text-accent-gold font-bold mt-0.5">②</span>
              <div>
                <strong>Adversarial bank + imperfect encoder:</strong> adv-training is required.
                Lifts retrieval by +71pp on V-STY/A-PARA, +17pp on A-SCN, +14pp on V-XC-ID.
              </div>
            </div>
            <div className="flex gap-3 items-start">
              <span className="text-accent-gold font-bold mt-0.5">③</span>
              <div>
                <strong>Clean encoder (cosine at ceiling):</strong> no headroom (A-XR-ID at RAG=1.00).
                AttMem matches RAG; training adds noise. Honest result.
              </div>
            </div>
          </div>
          <p className="mt-4 text-sm italic text-gray-600">
            The mechanism's value is measurable exactly where encoder cosine is imperfect on the
            deployment population.
          </p>
        </div>
      </div>
    </section>
  );
}

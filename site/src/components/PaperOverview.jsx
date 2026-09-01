const stages = [
  {
    number: '01',
    title: 'Ground what the user means',
    body: 'A vision-language or audio-language model resolves the referenced person, object, region, or time span in context.',
  },
  {
    number: '02',
    title: 'Identify who or what it is',
    body: 'A frozen specialist encoder extracts a cross-condition identity key from the grounded perception.',
  },
  {
    number: '03',
    title: 'Read it inside the model',
    body: 'A key/value row maps the perception to the language model’s own marker token, read by attention during generation.',
  },
];

const findings = [
  {
    value: '0.11×',
    title: 'Caption retention',
    body: 'On the least nameable signals, caption-based re-identification retains as little as 0.11 of a dedicated encoder’s recall.',
  },
  {
    value: '0.96',
    title: 'Grounded recall',
    body: 'Grounding plus a dedicated encoder reaches the correct-region oracle on the two-person face task; whole-scene encoding reaches 0.05.',
  },
  {
    value: '1,080',
    title: 'PerceptMem tasks',
    body: 'The benchmark spans 12 dataset/encoder domains across five perceptual modalities.',
  },
  {
    value: '10',
    title: 'Frozen model families',
    body: 'The training-free in-model read reproduces the encoder’s recall across transformer, untied-embedding, and hybrid-Mamba hosts.',
  },
];

export function PaperOverview() {
  return (
    <>
      <section id="approach" className="py-16 px-6 bg-gray-50 border-y border-gray-100 scroll-mt-20">
        <div className="max-w-6xl mx-auto">
          <p className="text-sm uppercase tracking-widest text-brand font-medium mb-3">Approach</p>
          <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark tracking-tight max-w-3xl">
            Separate grounding from identification, then give perception an in-model home.
          </h2>
          <div className="grid md:grid-cols-3 gap-5 mt-9">
            {stages.map((stage) => (
              <article key={stage.number} className="bg-white rounded-xl border border-gray-200 p-6">
                <div className="font-mono text-sm text-brand mb-4">{stage.number}</div>
                <h3 className="font-serif text-xl font-bold text-brand-dark mb-3">{stage.title}</h3>
                <p className="text-gray-600 leading-relaxed">{stage.body}</p>
              </article>
            ))}
          </div>
          <div className="mt-8 bg-white rounded-2xl shadow-sm p-4 md:p-6 border border-gray-200">
            <img
              src={`${import.meta.env.BASE_URL}figs/fig0_arch.png`}
              alt="Architecture: ground the referent, extract a perceptual identity key, and read its marker token from an in-model memory bank"
              className="w-full"
            />
          </div>
        </div>
      </section>

      <section id="evidence" className="py-16 px-6 scroll-mt-20">
        <div className="max-w-6xl mx-auto">
          <p className="text-sm uppercase tracking-widest text-brand font-medium mb-3">Evidence</p>
          <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark tracking-tight max-w-3xl">
            The perceptual signal survives when every component does the job it is good at.
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5 mt-9">
            {findings.map((finding) => (
              <article key={finding.title} className="rounded-xl border border-gray-200 p-6 bg-white">
                <div className="font-serif text-4xl font-bold text-brand mb-2">{finding.value}</div>
                <h3 className="font-bold text-brand-dark mb-2">{finding.title}</h3>
                <p className="text-sm text-gray-600 leading-relaxed">{finding.body}</p>
              </article>
            ))}
          </div>
          <div className="mt-10 grid md:grid-cols-2 gap-6">
            <div className="rounded-xl bg-brand-dark text-paper p-7">
              <h3 className="font-serif text-2xl font-bold mb-3">The encoder sets the ceiling</h3>
              <p className="text-gray-200 leading-relaxed">
                The recognition core is deliberately training-free. It preserves the specialist encoder’s quality while adding grounding, native in-model access, and O(1) registration.
              </p>
            </div>
            <div className="rounded-xl bg-gray-50 border border-gray-200 p-7">
              <h3 className="font-serif text-2xl font-bold text-brand-dark mb-3">Two memories compose</h3>
              <p className="text-gray-700 leading-relaxed">
                Perceptual identity belongs in the parametric bank; exact facts remain in a text store. Together they let an agent remember both what a user said and what they are like.
              </p>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

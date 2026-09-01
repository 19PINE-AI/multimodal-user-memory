import { motion } from 'framer-motion';
import { XCircle, AlertTriangle, Info, Clock, CheckCircle2 } from 'lucide-react';

const approaches = [
  {
    icon: XCircle, accent: 'text-rose-600', bg: 'bg-rose-50/50 border-rose-200',
    name: 'Text RAG',
    refs: 'Mem0, MemoryLLM',
    body: 'Caption each voice into text (Whisper + speaker descriptor): "A male voice, mid-30s, slight Eastern-European accent." Embed with a sentence encoder; cosine-retrieve.',
    verdict: 'Almost always fails. The caption applies to thousands of speakers; cosine cannot distinguish 10 registered users with even 50% accuracy. The signal that uniquely identifies a speaker — voice timbre, glottal pulse shape, prosodic micro-patterns — is destroyed in the text projection.',
  },
  {
    icon: AlertTriangle, accent: 'text-amber-600', bg: 'bg-amber-50/50 border-amber-200',
    name: 'Tool call',
    refs: 'M3-Agent',
    body: 'Call a face/speaker recognition tool (ArcFace, ECAPA-TDNN), receive a text label ("speaker 47"), store that.',
    verdict: 'The user\'s memory of speaker 47 is now just a text token. The perceptual fingerprint is never integrated into the LM\'s representation. At inference the LM can recall the label was assigned, but cannot reason about how the voice sounded.',
  },
  {
    icon: Info, accent: 'text-brand', bg: 'bg-blue-50/50 border-blue-200',
    name: 'Embedding RAG',
    refs: 'strongest text-free baseline',
    body: 'Skip captioning: index the raw encoder embedding (ECAPA, ArcFace, CLIP); cosine-retrieve at inference.',
    verdict: 'Succeeds when the encoder is perfectly cross-condition invariant; fails when it is not. RAG cosine reaches 0.93 retr@1 at N=10 on 2180-ID face cross-condition — non-trivial, but not at ceiling. We use this as the principled baseline.',
  },
  {
    icon: Clock, accent: 'text-gray-600', bg: 'bg-gray-50 border-gray-200',
    name: 'Per-concept gradient training',
    refs: 'MyVLM, Yo\'LLaVA',
    body: 'Train a per-concept token or adapter via SGD on registration samples.',
    verdict: 'Insertion cost is ~1 s per identity; the resulting memory is concept-specific, not modality-general. Doesn\'t compose to user-scale memory of dozens of perceptual characteristics.',
  },
  {
    icon: CheckCircle2, accent: 'text-accent-green', bg: 'bg-green-50/50 border-green-200',
    name: 'Parametric multimodal user memory',
    refs: 'this paper',
    body: 'Store the encoder embedding as key and the LM\'s value-side embedding (for an assigned marker token) as value. Cross-attention over the bank produces a residual injected at lm_head pre-forward, biasing the next-token logit toward the matching marker.',
    verdict: 'The LM gets a content-addressable memory in its own representation space, with no captioning intermediate. Specialises kNN-LM and Memorizing Transformers — attention over a non-parametric bank — to per-modality user content, with O(1) insertion.',
  },
];

export function WhyTextFails() {
  return (
    <section id="why-text-fails" className="py-16 px-6 bg-paper">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-serif text-3xl md:text-4xl font-bold text-brand-dark mb-3 tracking-tight">
          Why text memory cannot solve cross-condition perceptual recall
        </h2>
        <p className="text-gray-700 max-w-3xl text-base leading-relaxed mb-3">
          Consider speaker-identity recall under cross-recording conditions (the <strong>A-XR-ID</strong>
          sub-modality). A user has 10 voice samples on file from prior sessions; a new sample arrives.
          Which of the 10 is it?
        </p>
        <p className="text-gray-600 max-w-3xl text-sm mb-8 italic">
          We walk through five approaches; only the last avoids the lossy text projection.
        </p>

        <div className="space-y-3">
          {approaches.map((a, i) => (
            <motion.div
              key={a.name}
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.05 }}
              className={`rounded-lg border p-5 ${a.bg}`}
            >
              <div className="flex items-start gap-3">
                <a.icon className={`${a.accent} flex-shrink-0 mt-0.5`} size={22} />
                <div className="flex-1">
                  <div className="flex items-baseline gap-3 mb-1 flex-wrap">
                    <h3 className="font-bold text-brand-dark">{a.name}</h3>
                    <span className="text-xs text-gray-500 italic">{a.refs}</span>
                  </div>
                  <p className="text-sm text-gray-700 mb-2 leading-relaxed">{a.body}</p>
                  <p className={`text-sm font-medium leading-relaxed ${a.accent}`}>{a.verdict}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-8 rounded-xl border-2 border-brand bg-blue-50 p-5">
          <p className="text-sm text-brand-dark font-medium">
            <strong>The structural claim:</strong> captionable and perceptual user content require different
            memory primitives. "My cat is named Bibi" belongs in a text vector store; <em>how Bibi looks
            across lighting and angle</em> does not. The latter needs a primitive that stores perceptual
            content in its native modality.
          </p>
        </div>
      </div>
    </section>
  );
}

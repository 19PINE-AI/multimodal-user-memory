export const PAPER = {
  title: 'Parametric Multimodal User Memory',
  subtitle: 'Storing What Captions Cannot Carry',
  arxivId: '2608.28609',
  abstractUrl: 'https://arxiv.org/abs/2608.28609',
  pdfUrl: 'https://arxiv.org/pdf/2608.28609',
  repositoryUrl: 'https://github.com/19PINE-AI/multimodal-user-memory',
  websiteUrl: 'https://01.me/research/multimodal-user-memory/',
  authors: [
    { name: 'Bojie Li', affiliation: 'Pine AI' },
    { name: 'Noah Shi', affiliation: 'University of Washington' },
  ],
  abstract: [
    `A personalized agent needs a user memory: a persistent model of who its user is. Today it is almost always text — transcripts and captions retrieved by similarity. This serves the captionable half of a person (“my cat is named Bibi”), but discards the perceptual half no caption can hold: how a voice sounds, how a face reads across age and lighting, how tired someone sounds. We measure this loss across five modalities: a strong caption-based re-identifier recovers as little as 0.11 of a dedicated encoder's recall, collapsing toward chance on non-nameable signals.`,
    `We instead ground perceptual memory in the model, decomposing recall into two subproblems: a vision-language model grounds the referent in context (what and where), and a dedicated encoder extracts an identity key (who), stored as one inline token read by attention at generation with no external round-trip. Neither suffices alone — the VLM identifies cross-age faces at only 0.54 recall where a face encoder reaches 0.81, and an ungrounded encoder recognizes a two-person-scene referent at 0.05 — yet together they reach correct-region oracle (0.96), generalizing to multi-speaker audio and video. The recognition core is training-free: it reproduces the encoder's recall on any frozen model at O(1) registration cost. On PerceptMem (12 domains, 1,080 tasks) perceptual identity is capacity-limited (recall ≈ min(1, k/M) of the encoder's ceiling) while exact facts are binding-limited: identity belongs in a parametric bank, facts in a text store. The two memories compose cleanly: an agent with both can remember not only what its user said, but also what they are like.`,
  ],
  bibtex: `@misc{li2026parametricmultimodalusermemory,
  title         = {Parametric Multimodal User Memory: Storing What Captions Cannot Carry},
  author        = {Bojie Li and Noah Shi},
  year          = {2026},
  eprint        = {2608.28609},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.28609}
}`,
};

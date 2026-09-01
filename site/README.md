# Companion website

Static React + Vite + Tailwind site for **Parametric Multimodal User Memory:
Storing What Captions Cannot Carry** ([arXiv:2608.28609](https://arxiv.org/abs/2608.28609)).

The publication metadata, abstract, canonical arXiv links, and BibTeX live in
`src/paper.js`. Update that file first if the paper record changes.

## Local development

Requires Node.js 20.19+ or 22.13+.

```bash
npm ci
npm run dev       # serve at http://localhost:5173
npm run lint
npm run build     # produce static dist/
npm run preview
```

## Sections

- **Hero**: title, authors, abstract, paper links, and architecture diagram
- **Key Results**: 7 multi-seed BEATS-RAG cells in two tables
- **Why Text Fails**: comparison of textual and perceptual memory
- **Method**: equations and the four critical design choices
- **PerceptMem Scorecard**: Recharts comparison across five sub-modalities
- **Scaling**: log-scale results on the V-XC-ID-XXXL face pool
- **Path A**: parametric-memory experiment details
- **Latency**: AttMem and RAG-with-context comparison
- **Training Matters**: training-regime analysis
- **Ablations**: method-component ablations
- **Adversarial** and **Pareto**: robustness results and interactive tradeoff chart
- **Cross-family**, **VLM**, and **Cross-modal**: generalization results
- **Mechanism**: attention visualization
- **Demo**: animated registration and recall walkthrough
- **Reproducibility**: commands, publication links, and final BibTeX

## Deployment

The site is published under
<https://01.me/research/multimodal-user-memory/>. The Vite base is relative so
the generated `dist/` bundle can be hosted at that subpath without rewriting
asset URLs.

The `dist/` directory is intentionally gitignored; deploy a fresh `npm run
build` output rather than committing generated assets.

## Publication links

- Abstract: <https://arxiv.org/abs/2608.28609>
- PDF: <https://arxiv.org/pdf/2608.28609>
- Code: <https://github.com/19PINE-AI/multimodal-user-memory>

## Stack

- React 19, Vite 7, and Tailwind 3
- Recharts
- Framer Motion
- Lucide React
- React Syntax Highlighter

# Companion website

Static React + Vite + Tailwind site for the paper
**Parametric Multimodal User Memory: Storing What Captions Cannot Carry**.

## Local development

```bash
npm install
npm run dev          # serve at http://localhost:5173
npm run build        # produce static dist/
npm run preview      # serve the built dist/
```

## Sections

- **Hero**: title + abstract + architecture diagram (`fig0_arch.png`)
- **Key Results**: 7 multi-seed BEATS-RAG cells (3 random + 4 adversarial) in two tables
- **Method**: equations + the four critical design choices
- **PerceptMem Scorecard**: Recharts bar chart, 5 sub-modalities at N=10
- **Scaling**: log-scale line chart on V-XC-ID-XXXL (2180-ID face pool), with BEATS marker
- **Adversarial**: bar chart of standard-vs-adv-training across modalities, +71pp highlights
- **Pareto** (interactive): sub-modality switcher, scatter plot showing modality-dependent sweet spot
- **Cross-family**: Qwen 3B/7B vs Llama-3.1-8B vs Mistral-7B
- **VLM**: Qwen2.5-VL key-value-space orthogonality table
- **Latency**: log-log chart, AttMem vs RAG-with-context (OOM at 10k)
- **Mechanism**: 3-panel attention visualization figure
- **Demo** (animated): register 10 celebrities by first name, animated query → marker recall
- **Reproducibility**: code snippets with syntax highlighting + BibTeX

## Deployment

The `dist/` directory is a static bundle deployable anywhere:

- **GitHub Pages**: push `dist/` to a `gh-pages` branch
- **Vercel / Netlify**: link the repo, set build cmd `npm run build`, output `dist`
- **Cloudflare Pages**: same

## Updating data

When new experiment results land in `../results/`, edit `src/data/results.js` to sync.
The components are pure data-driven (Recharts).

## Stack

- React 19 + Vite 5 + Tailwind 3
- Recharts (data visualization)
- Framer Motion (entrance animations)
- Lucide React (icons)
- React Syntax Highlighter (code snippets)

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

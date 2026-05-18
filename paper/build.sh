#!/bin/bash
# Reproducible paper build: figs + pdflatex + bibtex + pdflatex (×2).
# pdflatex returns rc=1 on harmless warnings, so we don't 'set -e'.
cd "$(dirname "$0")"
python3 figs/generate_figs.py 2>&1 | grep -v "^/home\|UserWarning\|warnings.warn\|^$"
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
if [ -f main.pdf ]; then
  echo "Built main.pdf ($(stat -c %s main.pdf) bytes, $(pdfinfo main.pdf | awk '/^Pages/{print $2}') pages)"
else
  echo "FAILED — no main.pdf produced; see main.log"; exit 1
fi

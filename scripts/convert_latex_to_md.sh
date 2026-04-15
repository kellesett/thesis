#!/usr/bin/env bash
# scripts/convert_latex_to_md.sh
#
# Для каждой папки в datasets/surge/latex_src/<arxiv_id>/:
#   pandoc --citeproc --bibliography file.bib --csl ieee.csl merged.tex -o merged.md
#
# Если .bib нет (inline thebibliography) — pandoc без --citeproc.
#
# Usage:
#   bash scripts/convert_latex_to_md.sh
#   bash scripts/convert_latex_to_md.sh 1911.02794

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/datasets/surge/latex_src"
CSL="$SRC_DIR/ieee.csl"

if ! command -v pandoc &>/dev/null; then
    echo "[ERROR] pandoc not found. Install with: brew install pandoc"
    exit 1
fi

if [[ ! -f "$CSL" ]]; then
    echo "[ERROR] ieee.csl not found at $CSL"
    exit 1
fi

process_one() {
    local arxiv_id="$1"
    local dir="$SRC_DIR/$arxiv_id"
    local merged="$dir/merged.tex"
    local out="$dir/merged.md"

    [[ ! -f "$merged" ]] && { echo "  [SKIP] $arxiv_id — no merged.tex"; return 0; }
    [[ -f "$out" ]]      && { echo "  [SKIP] $arxiv_id — already exists"; return 0; }

    # Найти .bib: сначала с тем же именем что главный .tex, иначе любой
    local main_tex stem bib_file=""
    main_tex="$(grep -rl '\\documentclass' "$dir" --include="*.tex" 2>/dev/null | head -1 || true)"
    if [[ -n "$main_tex" ]]; then
        stem="$(basename "$main_tex" .tex)"
        [[ -f "$dir/${stem}.bib" ]] && bib_file="$dir/${stem}.bib"
    fi
    [[ -z "$bib_file" ]] && bib_file="$(find "$dir" -maxdepth 2 -name "*.bib" | head -1 || true)"

    local err
    if [[ -n "$bib_file" ]]; then
        if pandoc --citeproc --bibliography "$bib_file" --csl "$CSL" "$merged" -o "$out" 2>/dev/null; then
            echo "  [OK]   $arxiv_id (bib: $(basename "$bib_file"))"
        else
            err="$(pandoc --citeproc --bibliography "$bib_file" --csl "$CSL" "$merged" -o "$out" 2>&1 | head -1)"
            echo "  [FAIL] $arxiv_id — $err"
            rm -f "$out"; return 1
        fi
    else
        # Нет .bib — pandoc без citeproc, inline thebibliography
        if pandoc "$merged" -o "$out" 2>/dev/null; then
            echo "  [OK]   $arxiv_id (no bib, inline)"
        else
            err="$(pandoc "$merged" -o "$out" 2>&1 | head -1)"
            echo "  [FAIL] $arxiv_id — $err"
            rm -f "$out"; return 1
        fi
    fi
}

if [[ $# -ge 1 ]]; then
    process_one "$1"
else
    total=0; ok=0; fail=0
    for dir in "$SRC_DIR"/*/; do
        arxiv_id="$(basename "$dir")"
        [[ "$arxiv_id" == "arxiv_cache.json" ]] && continue
        [[ ! -d "$dir" ]] && continue
        ((total++)) || true
        if process_one "$arxiv_id"; then ((ok++)) || true
        else ((fail++)) || true; fi
    done
    echo ""
    echo "Done: $ok ok, $fail failed, $total total"
fi

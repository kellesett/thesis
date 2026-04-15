#!/usr/bin/env bash
# scripts/merge_latex.sh
#
# Для каждой папки в datasets/surge/latex_src/<arxiv_id>/:
#   1. Находит главный .tex (содержит \documentclass)
#   2. Запускает latexpand для раскрытия \input{}/\include{}
#   3. Вставляет .bbl вместо \bibliography{...}
#   4. Сохраняет результат в merged.tex
#
# Usage:
#   bash scripts/merge_latex.sh
#   bash scripts/merge_latex.sh 1911.02794   # одна статья

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/datasets/surge/latex_src"

# Проверяем latexpand
if ! command -v latexpand &>/dev/null; then
    echo "[ERROR] latexpand not found. Install with: sudo tlmgr install latexpand"
    exit 1
fi

process_one() {
    local arxiv_id="$1"
    local dir="$SRC_DIR/$arxiv_id"
    local out="$dir/merged.tex"

    # Уже обработан
    if [[ -f "$out" ]]; then
        echo "  [SKIP] $arxiv_id — merged.tex already exists"
        return 0
    fi

    # Найти главный .tex (содержит \documentclass)
    local main_tex=""
    while IFS= read -r f; do
        if grep -q '\\documentclass' "$f" 2>/dev/null; then
            main_tex="$f"
            break
        fi
    done < <(find "$dir" -maxdepth 2 -name "*.tex" | sort)

    if [[ -z "$main_tex" ]]; then
        echo "  [WARN] $arxiv_id — no main .tex found (no \\documentclass)"
        return 1
    fi

    local main_name
    main_name="$(basename "$main_tex")"

    # Найти .bbl: сначала с тем же именем что .tex, иначе любой
    local stem bbl_file
    stem="$(basename "$main_tex" .tex)"
    if [[ -f "$dir/${stem}.bbl" ]]; then
        bbl_file="$dir/${stem}.bbl"
    else
        bbl_file="$(find "$dir" -maxdepth 2 -name "*.bbl" | head -1)"
    fi

    # Проверяем есть ли активная (не закомментированная) \bibliography{} в тексте
    local has_active_bib=false
    if grep -qE '^[^%]*\\bibliography\s*\{' "$main_tex" 2>/dev/null; then
        has_active_bib=true
    fi

    # latexpand: раскрыть \input{}/\include{} и вставить .bbl вместо \bibliography{}
    if [[ -n "$bbl_file" && "$has_active_bib" == "true" ]]; then
        latexpand --expand-bbl "$bbl_file" "$main_tex" > "$out" 2>/dev/null || {
            echo "  [WARN] $arxiv_id — latexpand failed on $main_name"
            return 1
        }
        echo "  [OK]   $arxiv_id — merged from $main_name + $(basename "$bbl_file")"
    else
        latexpand "$main_tex" > "$out" 2>/dev/null || {
            echo "  [WARN] $arxiv_id — latexpand failed on $main_name"
            return 1
        }
        if [[ -n "$bbl_file" && "$has_active_bib" == "false" ]]; then
            echo "  [WARN] $arxiv_id — \\bibliography is commented out, .bbl skipped: $main_name"
        else
            echo "  [OK]   $arxiv_id — merged from $main_name (inline bib)"
        fi
    fi

    # Если нет .bib — генерируем из inline bibliography в merged.tex
    local bib_file=""
    if [[ -f "$dir/${stem}.bib" ]]; then
        bib_file="$dir/${stem}.bib"
    else
        bib_file="$(find "$dir" -maxdepth 2 -name "*.bib" ! -name "generated.bib" | head -1)"
    fi
    if [[ -z "$bib_file" ]]; then
        python3 "$ROOT/scripts/bbl_to_bib.py" "$out" "$dir/generated.bib" 2>/dev/null && \
            echo "  [INFO] $arxiv_id — generated.bib from inline bibliography" || \
            echo "  [WARN] $arxiv_id — could not generate .bib from inline bibliography"
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────────

if [[ $# -ge 1 ]]; then
    # Обработать конкретную статью
    process_one "$1"
else
    # Обработать все папки
    total=0
    ok=0
    fail=0

    for dir in "$SRC_DIR"/*/; do
        arxiv_id="$(basename "$dir")"
        [[ "$arxiv_id" == "arxiv_cache.json" ]] && continue
        [[ ! -d "$dir" ]] && continue

        ((total++)) || true
        if process_one "$arxiv_id"; then
            ((ok++)) || true
        else
            ((fail++)) || true
        fi
    done

    echo ""
    echo "Done: $ok ok, $fail failed, $total total"
fi

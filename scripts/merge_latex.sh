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
#   bash scripts/merge_latex.sh                 # все папки, skip существующих
#   bash scripts/merge_latex.sh --force         # пересобрать всё
#   bash scripts/merge_latex.sh 1911.02794      # одна статья
#   bash scripts/merge_latex.sh --force 1911.02794
#
# ВАЖНО: latexpand вызывается из каталога статьи (cd "$dir" && latexpand "$basename"),
# иначе относительные \input{} внутри .tex резолвятся относительно текущего каталога
# вызова скрипта и не находятся.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/datasets/surge/latex_src"

FORCE=false

# Проверяем latexpand
if ! command -v latexpand &>/dev/null; then
    echo "[ERROR] latexpand not found. Install with: sudo tlmgr install latexpand"
    exit 1
fi

process_one() {
    local arxiv_id="$1"
    local dir="$SRC_DIR/$arxiv_id"
    local out="$dir/merged.tex"

    if [[ ! -d "$dir" ]]; then
        echo "  [WARN] $arxiv_id — directory not found: $dir"
        return 1
    fi

    # Уже обработан
    if [[ -f "$out" && "$FORCE" != "true" ]]; then
        echo "  [SKIP] $arxiv_id — merged.tex already exists (use --force to rebuild)"
        return 0
    fi

    # При --force удаляем старый merged.tex ДО поиска main_tex, чтобы он не был
    # ошибочно выбран find'ом как кандидат (в нём есть \documentclass).
    if [[ "$FORCE" == "true" && -f "$out" ]]; then
        rm -f "$out"
    fi

    # Найти главный .tex (содержит \documentclass).
    # Исключаем наш собственный merged.tex на случай, если он остался от прошлых
    # прогонов без --force и пайплайн всё равно вызывается повторно.
    local main_tex=""
    while IFS= read -r f; do
        [[ "$(basename "$f")" == "merged.tex" ]] && continue
        if grep -q '\\documentclass' "$f" 2>/dev/null; then
            main_tex="$f"
            break
        fi
    done < <(find "$dir" -maxdepth 2 -name "*.tex" | sort)

    if [[ -z "$main_tex" ]]; then
        echo "  [WARN] $arxiv_id — no main .tex found (no \\documentclass)"
        return 1
    fi

    # Каталог, в котором лежит main_tex (может быть поддиректорией $dir)
    local main_dir main_name
    main_dir="$(dirname "$main_tex")"
    main_name="$(basename "$main_tex")"

    # Найти .bbl: сначала с тем же именем что .tex, иначе любой
    local stem bbl_file bbl_rel
    stem="$(basename "$main_tex" .tex)"
    bbl_file=""
    if [[ -f "$main_dir/${stem}.bbl" ]]; then
        bbl_file="$main_dir/${stem}.bbl"
    else
        bbl_file="$(find "$dir" -maxdepth 2 -name "*.bbl" | head -1)"
    fi

    # Относительный путь к .bbl от main_dir (latexpand запускается из main_dir)
    bbl_rel=""
    if [[ -n "$bbl_file" ]]; then
        if command -v realpath &>/dev/null; then
            bbl_rel="$(realpath --relative-to="$main_dir" "$bbl_file" 2>/dev/null || echo "")"
        fi
        # Фолбэк через python, если realpath --relative-to не поддерживается (macOS без coreutils)
        if [[ -z "$bbl_rel" ]]; then
            bbl_rel="$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$bbl_file" "$main_dir")"
        fi
    fi

    # Проверяем есть ли активная (не закомментированная) \bibliography{} в тексте
    local has_active_bib=false
    if grep -qE '^[^%]*\\bibliography\s*\{' "$main_tex" 2>/dev/null; then
        has_active_bib=true
    fi

    # latexpand: раскрыть \input{}/\include{} и вставить .bbl вместо \bibliography{}
    # ВАЖНО: cd в каталог main_tex, чтобы относительные \input{} резолвились верно.
    local tmp_out="$out.tmp"
    if [[ -n "$bbl_file" && "$has_active_bib" == "true" ]]; then
        if ( cd "$main_dir" && latexpand --expand-bbl "$bbl_rel" "$main_name" ) > "$tmp_out" 2>/dev/null; then
            mv "$tmp_out" "$out"
            echo "  [OK]   $arxiv_id — merged from $main_name + $(basename "$bbl_file")"
        else
            rm -f "$tmp_out"
            echo "  [WARN] $arxiv_id — latexpand failed on $main_name"
            return 1
        fi
    else
        if ( cd "$main_dir" && latexpand "$main_name" ) > "$tmp_out" 2>/dev/null; then
            mv "$tmp_out" "$out"
            if [[ -n "$bbl_file" && "$has_active_bib" == "false" ]]; then
                echo "  [WARN] $arxiv_id — \\bibliography is commented out, .bbl skipped: $main_name"
            else
                echo "  [OK]   $arxiv_id — merged from $main_name (inline bib)"
            fi
        else
            rm -f "$tmp_out"
            echo "  [WARN] $arxiv_id — latexpand failed on $main_name"
            return 1
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

# Парсим аргументы: --force и опциональный arxiv_id
TARGET=""
for arg in "$@"; do
    case "$arg" in
        --force|-f)
            FORCE=true
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            TARGET="$arg"
            ;;
    esac
done

if [[ -n "$TARGET" ]]; then
    # Обработать конкретную статью
    process_one "$TARGET"
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

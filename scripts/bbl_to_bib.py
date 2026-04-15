#!/usr/bin/env python3
"""
scripts/bbl_to_bib.py

Генерирует минимальный .bib из inline \begin{thebibliography} в .tex файле.
Используется когда .bib отсутствует в тарболле.

Для каждого \bibitem{key} пытается извлечь title:
  - Ищет текст в LaTeX-кавычках ``...''
  - Если не найден — title = None (записывается как пустая строка в .bib)

Usage:
    python scripts/bbl_to_bib.py <merged.tex> <output.bib>
"""
import re
import sys
from pathlib import Path


def extract_bibitems(tex: str) -> list[tuple[str, str]]:
    """Извлекает список (key, raw_text) из \begin{thebibliography}...\end{thebibliography}."""
    m = re.search(
        r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
        tex, re.DOTALL
    )
    if not m:
        return []

    bib_block = m.group(0)

    # Разбиваем по \bibitem{key}
    parts = re.split(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', bib_block)
    # parts = [preamble, key1, text1, key2, text2, ...]

    items = []
    it = iter(parts[1:])  # пропускаем преамбулу
    for key, raw in zip(it, it):
        items.append((key.strip(), raw.strip()))
    return items


def extract_title(raw: str) -> str | None:
    """Пытается извлечь title из отформатированного bibitem текста."""
    # Убираем LaTeX-команды типа \BIBentryALTinterwordspacing и т.п.
    raw = re.sub(r'\\BIBentry\w+', '', raw)
    raw = re.sub(r'\\newblock\b', ' ', raw)

    # Вариант 1: LaTeX-кавычки ``title,'' или ``title.''
    m = re.search(r'``(.+?)(?:,\'\'|\.\'\'|\'\')', raw, re.DOTALL)
    if m:
        title = m.group(1).strip()
        title = _clean_latex(title)
        if len(title) > 5:
            return title

    # Вариант 2: кавычки "title" (unicode)
    m = re.search(r'"(.+?)"', raw, re.DOTALL)
    if m:
        title = m.group(1).strip()
        title = _clean_latex(title)
        if len(title) > 5:
            return title

    # Вариант 3: \emph{title} (некоторые форматы)
    m = re.search(r'\\emph\{([^}]{10,})\}', raw)
    if m:
        title = _clean_latex(m.group(1).strip())
        if len(title) > 5:
            return title

    return None


def _clean_latex(s: str) -> str:
    """Убирает основные LaTeX-команды из строки."""
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)  # \cmd{text} → text
    s = re.sub(r'\\[a-zA-Z]+\b', ' ', s)              # \cmd → пробел
    s = re.sub(r'[{}]', '', s)                          # фигурные скобки
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def bbl_to_bib(tex_path: Path, bib_path: Path) -> int:
    """Конвертирует inline bibliography из tex_path в bib_path.
    Возвращает количество записей."""
    tex = tex_path.read_text(encoding='utf-8', errors='replace')
    items = extract_bibitems(tex)

    if not items:
        print(f"[bbl_to_bib] No \\begin{{thebibliography}} found in {tex_path}", flush=True)
        return 0

    lines = []
    for key, raw in items:
        title = extract_title(raw)
        # Экранируем фигурные скобки в title для BibTeX
        if title:
            title_bib = title.replace('{', '{{').replace('}', '}}')
        else:
            title_bib = ""  # pandoc сможет разрешить ключ, title будет пустым

        lines.append(f"@misc{{{key},")
        lines.append(f"  title = {{{title_bib}}}")
        lines.append("}")
        lines.append("")

    bib_path.write_text('\n'.join(lines), encoding='utf-8')
    return len(items)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <merged.tex> <output.bib>")
        sys.exit(1)

    tex_path = Path(sys.argv[1])
    bib_path = Path(sys.argv[2])

    n = bbl_to_bib(tex_path, bib_path)
    print(f"[bbl_to_bib] {n} entries → {bib_path}")

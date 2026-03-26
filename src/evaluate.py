"""
Оценка сгенерированных survey через LLM-as-Judge по методологии SurveyForge.

Два режима оценки (включаются флагами):
  --swr   Score Win Rate      — судья независимо оценивает каждый текст (0-100),
                                победитель = у кого балл выше
  --cwr   Comparative Win Rate — судья видит оба текста рядом и выбирает лучший напрямую

Если ни один флаг не передан — ничего не считается.

Что оценивается:
  --eval outline   — сравнение структуры/оглавления (быстро, дёшево)
  --eval content   — сравнение полного текста (дорого, медленно)
  --eval outline,content — оба

Конфиг судей — JSON-файл (--judges judges.json):
[
  {
    "name": "claude",                             # произвольное имя для логов
    "url": "https://openrouter.ai/api/v1",
    "model": "anthropic/claude-sonnet-4-5",
    "api_key_env": "OPENROUTER_API_KEY",          # имя переменной из .env
    "n": 1                                        # сколько раз звать эту модель
  },
  {
    "name": "gemini_flash",
    "url": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.5-flash",
    "api_key_env": "OPENROUTER_API_KEY",
    "n": 2                                        # 2 прогона → усредняем
  },
  {
    "name": "local",
    "url": "http://localhost:8000/v1",
    "model": "deepseek-chat",
    "api_key_env": "LOCAL_API_KEY",
    "n": 1
  }
]

Примеры запуска:
  # Score Win Rate по outline, судья — Gemini Flash
  python 03_evaluate.py --swr --eval outline --judges judges.json

  # Оба метода по контенту, судья — Claude
  python 03_evaluate.py --swr --cwr --eval content --judges judges_claude.json

  # Всё сразу (дорого!)
  python 03_evaluate.py --swr --cwr --eval outline,content --judges judges.json

  # Resume: пропустить уже посчитанные
  python 03_evaluate.py --swr --eval outline --judges judges.json --resume
"""

import os
import json
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR      = Path(__file__).parent.parent
RESULTS_DIR   = ROOT_DIR / "results"
DATASETS_DIR  = ROOT_DIR / "datasets" / "human_surveys"
EVAL_OUT_DIR  = ROOT_DIR / "eval_results"

# Максимальная длина текста передаваемого судье (в символах)
# outline — короткий, content — обрезаем до разумного размера
MAX_OUTLINE_CHARS = 3_000
MAX_CONTENT_CHARS = 12_000


# ─── Конфиг судей ─────────────────────────────────────────────────────────────

@dataclass
class JudgeConfig:
    name: str
    url: str
    model: str
    api_key_env: str
    n: int = 1          # сколько раз вызывать

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise EnvironmentError(
                f"Переменная {self.api_key_env} не задана в .env (судья '{self.name}')"
            )
        return key

    def make_client(self) -> OpenAI:
        return OpenAI(base_url=self.url, api_key=self.api_key)


def load_judges(path: str) -> list[JudgeConfig]:
    with open(path) as f:
        raw = json.load(f)
    judges = []
    for item in raw:
        judges.append(JudgeConfig(
            name=item["name"],
            url=item["url"],
            model=item["model"],
            api_key_env=item["api_key_env"],
            n=item.get("n", 1),
        ))
    return judges


# ─── Промпты ──────────────────────────────────────────────────────────────────

# --- Score Win Rate ---

SWR_PROMPT = """\
You are evaluating a scientific survey on the topic: "{topic}"

{eval_type_desc}

{item_label}:
<text>
{text}
</text>

Score this survey from 0 to 100 based on the following criteria:
- **Structure**: logical organization, clear hierarchy, well-defined sections
- **Comprehensiveness**: coverage of key subfields, important papers, and milestones
- **Coherence**: smooth transitions, consistent terminology, well-integrated content

Think step by step, then give a final score.

Format your answer as:
REASONING: <brief analysis>
SCORE: <integer 0-100>\
"""

# --- Comparative Win Rate ---

CWR_PROMPT = """\
You are comparing two scientific surveys on the topic: "{topic}"

{eval_type_desc}

Survey A:
<survey_a>
{text_a}
</survey_a>

Survey B:
<survey_b>
{text_b}
</survey_b>

Compare both surveys on:
- **Structure**: logical organization, clear hierarchy, well-defined sections
- **Comprehensiveness**: coverage of key subfields, important papers, and milestones
- **Coherence**: smooth transitions, consistent terminology, well-integrated content

Think step by step, then give your verdict.

Format your answer as:
REASONING: <brief analysis>
WINNER: <A or B or TIE>\
"""

EVAL_TYPE_DESCS = {
    "outline": "You are evaluating the OUTLINE ONLY (section and subsection titles, structure).",
    "content": "You are evaluating the FULL TEXT of the survey.",
}


# ─── Структуры результатов ────────────────────────────────────────────────────

@dataclass
class JudgeCall:
    judge_name: str
    judge_model: str
    call_index: int       # 0-based, если n > 1
    raw_response: str
    # для SWR
    score: Optional[int] = None
    # для CWR
    winner: Optional[str] = None  # "A", "B", "TIE"
    error: Optional[str] = None


@dataclass
class PairResult:
    """Результат оценки одной пары (system, topic) одним методом."""
    system_id: str
    topic: str
    eval_type: str        # "outline" или "content"
    method: str           # "swr" или "cwr"
    judge_calls: list[JudgeCall] = field(default_factory=list)

    # Агрегированные по всем вызовам всех судей
    @property
    def verdict(self) -> str:
        """win / lose / tie / unknown"""
        if self.method == "swr":
            scores = [(c.score, c.score) for c in self.judge_calls
                      if c.score is not None]
            # у нас нет human score в judge_calls — смотрим ниже
            return "unknown"  # агрегация делается в _aggregate_swr
        if self.method == "cwr":
            verdicts = [c.winner for c in self.judge_calls if c.winner]
            if not verdicts:
                return "unknown"
            wins  = verdicts.count("A")   # A = generated
            loses = verdicts.count("B")   # B = human
            ties  = verdicts.count("TIE")
            if wins > loses and wins > ties:
                return "win"
            if loses > wins and loses > ties:
                return "lose"
            return "tie"
        return "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SWRPairResult:
    """SWR нужен отдельный класс — храним два набора оценок."""
    system_id: str
    topic: str
    eval_type: str
    generated_calls: list[JudgeCall] = field(default_factory=list)
    human_calls: list[JudgeCall] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        gen_scores  = [c.score for c in self.generated_calls  if c.score is not None]
        hum_scores  = [c.score for c in self.human_calls       if c.score is not None]
        if not gen_scores or not hum_scores:
            return "unknown"
        gen_avg = sum(gen_scores) / len(gen_scores)
        hum_avg = sum(hum_scores) / len(hum_scores)
        if gen_avg > hum_avg + 1:
            return "win"
        if gen_avg < hum_avg - 1:
            return "lose"
        return "tie"

    @property
    def generated_avg_score(self) -> Optional[float]:
        scores = [c.score for c in self.generated_calls if c.score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def human_avg_score(self) -> Optional[float]:
        scores = [c.score for c in self.human_calls if c.score is not None]
        return sum(scores) / len(scores) if scores else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        d["generated_avg_score"] = self.generated_avg_score
        d["human_avg_score"] = self.human_avg_score
        return d


# ─── Утилиты ──────────────────────────────────────────────────────────────────

def call_judge(client: OpenAI, model: str, prompt: str,
               max_tokens: int = 1000) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
        extra_headers={
            "HTTP-Referer": "https://github.com/survey-benchmark",
            "X-Title": "Survey Evaluation",
        },
    )
    return resp.choices[0].message.content or ""


def parse_score(text: str) -> Optional[int]:
    m = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return max(0, min(100, int(m.group(1))))
    nums = re.findall(r"\b(\d{1,3})\b", text)
    if nums:
        return max(0, min(100, int(nums[-1])))
    return None


def parse_winner(text: str) -> Optional[str]:
    m = re.search(r"WINNER:\s*(A|B|TIE)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # fallback
    if re.search(r"\bSurvey A\b.*\bbetter\b", text, re.IGNORECASE):
        return "A"
    if re.search(r"\bSurvey B\b.*\bbetter\b", text, re.IGNORECASE):
        return "B"
    return None


def extract_outline(text: str) -> str:
    """Извлекает заголовки секций из markdown-текста."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            lines.append(s)
    result = "\n".join(lines)
    return result[:MAX_OUTLINE_CHARS] if result else text[:MAX_OUTLINE_CHARS]


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... truncated at {max_chars} chars ...]"


# ─── Загрузка данных ──────────────────────────────────────────────────────────

def load_generated_results(topic: str) -> dict[str, dict]:
    """
    Загружает все сгенерированные survey для данной темы.
    Ключ — system_id, значение — словарь из JSON-файла.
    """
    results = {}
    if not RESULTS_DIR.exists():
        return results
    # slug темы: то же что в base.py
    topic_slug = re.sub(r"[^\w]", "_", topic).strip("_")
    for f in RESULTS_DIR.glob(f"*__{topic_slug}.json"):
        sys_id = f.stem.split("__")[0]
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data.get("generated_text"):
                results[sys_id] = data
        except Exception:
            pass
    return results


def load_human_survey(topic: str) -> Optional[dict]:
    """
    Загружает human survey для темы из datasets/human_surveys/{topic}.json
    Возвращает dict с ключами: text, outline (или None если нет файла).
    """
    safe = topic.replace("/", "_").replace("\\", "_")
    path = DATASETS_DIR / f"{safe}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # берём первый survey (обычно один на тему)
    surveys = data.get("surveys", [])
    if not surveys:
        return None
    return surveys[0]


def get_text_for_eval(survey_data: dict, eval_type: str) -> str:
    """Возвращает нужный кусок текста в зависимости от типа оценки."""
    text = survey_data.get("text", "") or survey_data.get("generated_text", "")
    outline = survey_data.get("outline", "")

    if eval_type == "outline":
        if outline:
            return truncate(outline, MAX_OUTLINE_CHARS)
        # Если нет готового outline — извлекаем из текста
        return extract_outline(text)

    # content
    return truncate(text, MAX_CONTENT_CHARS)


# ─── Оценка ───────────────────────────────────────────────────────────────────

def run_swr(judges: list[JudgeConfig], topic: str,
            gen_text: str, human_text: str,
            eval_type: str, system_id: str) -> SWRPairResult:
    """Score Win Rate: оцениваем generated и human независимо, сравниваем баллы."""
    result = SWRPairResult(system_id=system_id, topic=topic, eval_type=eval_type)
    desc = EVAL_TYPE_DESCS[eval_type]

    for judge in judges:
        client = judge.make_client()
        for i in range(judge.n):
            # Оцениваем generated
            prompt_gen = SWR_PROMPT.format(
                topic=topic,
                eval_type_desc=desc,
                item_label="Generated survey",
                text=gen_text,
            )
            try:
                resp = call_judge(client, judge.model, prompt_gen)
                score = parse_score(resp)
                result.generated_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response=resp, score=score,
                ))
            except Exception as e:
                result.generated_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response="", error=str(e),
                ))
            time.sleep(0.5)

            # Оцениваем human
            prompt_hum = SWR_PROMPT.format(
                topic=topic,
                eval_type_desc=desc,
                item_label="Human-written survey",
                text=human_text,
            )
            try:
                resp = call_judge(client, judge.model, prompt_hum)
                score = parse_score(resp)
                result.human_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response=resp, score=score,
                ))
            except Exception as e:
                result.human_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response="", error=str(e),
                ))
            time.sleep(0.5)

    return result


def run_cwr(judges: list[JudgeConfig], topic: str,
            gen_text: str, human_text: str,
            eval_type: str, system_id: str) -> PairResult:
    """Comparative Win Rate: судья сравнивает оба текста напрямую.
    Generated = Survey A, Human = Survey B.
    """
    result = PairResult(system_id=system_id, topic=topic,
                        eval_type=eval_type, method="cwr")
    desc = EVAL_TYPE_DESCS[eval_type]

    for judge in judges:
        client = judge.make_client()
        for i in range(judge.n):
            prompt = CWR_PROMPT.format(
                topic=topic,
                eval_type_desc=desc,
                text_a=gen_text,
                text_b=human_text,
            )
            try:
                resp = call_judge(client, judge.model, prompt, max_tokens=800)
                winner = parse_winner(resp)
                result.judge_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response=resp, winner=winner,
                ))
            except Exception as e:
                result.judge_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response="", error=str(e),
                ))
            time.sleep(0.5)

    return result


# ─── Win Rate агрегация ───────────────────────────────────────────────────────

def compute_win_rate(verdicts: list[str]) -> dict:
    """
    Win rate по формуле SurveyForge: (wins + 0.5 * ties) / total
    Возвращает также wins, ties, loses, total.
    """
    wins  = verdicts.count("win")
    loses = verdicts.count("lose")
    ties  = verdicts.count("tie")
    total = wins + loses + ties
    wr = (wins + 0.5 * ties) / total if total else 0.0
    return {"win_rate": wr, "wins": wins, "ties": ties,
            "loses": loses, "total": total}


# ─── CLI и точка входа ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation (Score WR + Comparative WR)"
    )
    # Методы оценки
    parser.add_argument("--swr", action="store_true",
                        help="Включить Score Win Rate")
    parser.add_argument("--cwr", action="store_true",
                        help="Включить Comparative Win Rate")
    # Что оцениваем
    parser.add_argument("--eval", default="outline",
                        help="Что оценивать: outline, content, или outline,content")
    # Конфиг судей
    parser.add_argument("--judges", default=str(ROOT_DIR / "configs" / "judges.json"),
                        help="Путь к JSON-файлу с конфигом судей (default: judges.json)")
    # Фильтры
    parser.add_argument("--systems", default=None,
                        help="Фильтр систем через запятую (default: все)")
    parser.add_argument("--topics", default=None,
                        help="Фильтр тем через запятую (default: все)")
    # Прочее
    parser.add_argument("--out", default=str(EVAL_OUT_DIR),
                        help=f"Папка для результатов (default: {EVAL_OUT_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Пропускать уже посчитанные пары")
    args = parser.parse_args()

    # Проверяем что хоть что-то включено
    if not args.swr and not args.cwr:
        parser.error("Укажи хотя бы один метод: --swr и/или --cwr")

    eval_types = [e.strip() for e in args.eval.split(",")]
    for et in eval_types:
        if et not in ("outline", "content"):
            parser.error(f"Неизвестный тип оценки: '{et}'. Допустимо: outline, content")

    # Загружаем судей
    judges_path = args.judges
    if not Path(judges_path).exists():
        parser.error(
            f"Файл судей не найден: {judges_path}\n"
            "Создай judges.json или укажи путь через --judges"
        )
    judges = load_judges(judges_path)
    print(f"Судьи ({len(judges)}):")
    for j in judges:
        print(f"  {j.name}: {j.model} @ {j.url}  (n={j.n})")

    # Фильтры
    sys_filter   = set(args.systems.split(",")) if args.systems else None
    topic_filter = set(args.topics.split(","))  if args.topics  else None

    # Собираем список тем
    import sys as _sys; _sys.path.insert(0, str(ROOT_DIR / "src"))
    from generate import load_topics, SURVEYBENCH_DIR
    all_topics = load_topics(n=10)
    if topic_filter:
        all_topics = [t for t in all_topics if t["topic"] in topic_filter]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = []
    if args.swr: methods.append("swr")
    if args.cwr: methods.append("cwr")

    print(f"\nМетоды: {methods}")
    print(f"Типы оценки: {eval_types}")
    print(f"Тем: {len(all_topics)}\n")

    # ── Главный цикл ──────────────────────────────────────────────────────────
    all_swr_results: dict[str, list[SWRPairResult]] = defaultdict(list)
    all_cwr_results: dict[str, list[PairResult]]    = defaultdict(list)

    for topic_data in tqdm(all_topics, desc="Темы"):
        topic = topic_data["topic"]

        # Загружаем human survey
        human = load_human_survey(topic)
        if human is None:
            tqdm.write(f"  [SKIP] Нет human survey для '{topic}'. "
                       f"Запусти download_dataset.py")
            continue

        # Загружаем сгенерированные результаты
        generated = load_generated_results(topic)
        if not generated:
            tqdm.write(f"  [SKIP] Нет сгенерированных результатов для '{topic}'. "
                       f"Запусти base.py")
            continue

        if sys_filter:
            generated = {k: v for k, v in generated.items() if k in sys_filter}

        for sys_id, gen_data in generated.items():
            for eval_type in eval_types:
                gen_text   = get_text_for_eval(gen_data, eval_type)
                human_text = get_text_for_eval(human,    eval_type)

                for method in methods:
                    out_file = out_dir / f"{sys_id}__{re.sub(r'[^\\w]','_',topic)}__{eval_type}__{method}.json"

                    if args.resume and out_file.exists():
                        tqdm.write(f"  [SKIP] {sys_id}/{topic[:30]}/{eval_type}/{method}")
                        continue

                    tqdm.write(f"  [{method.upper()}|{eval_type}] {sys_id} / {topic[:45]}")

                    if method == "swr":
                        res = run_swr(judges, topic, gen_text, human_text,
                                      eval_type, sys_id)
                        verdict = res.verdict
                        score_info = (f"gen={res.generated_avg_score:.1f} "
                                      f"hum={res.human_avg_score:.1f}"
                                      if res.generated_avg_score is not None else "no scores")
                        tqdm.write(f"    → {verdict} ({score_info})")
                        all_swr_results[sys_id].append(res)
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)

                    elif method == "cwr":
                        res = run_cwr(judges, topic, gen_text, human_text,
                                      eval_type, sys_id)
                        verdict = res.verdict
                        tqdm.write(f"    → {verdict}")
                        all_cwr_results[sys_id].append(res)
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)

    # ── Итоговые таблицы ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")

    if args.swr and all_swr_results:
        print("\nScore Win Rate (SWR):")
        print(f"{'Система':<28} {'Тип':>8} {'WR':>6} {'Gen':>6} {'Hum':>6} {'W/T/L':>10}")
        print("-" * 70)
        for sys_id, results in sorted(all_swr_results.items()):
            for et in eval_types:
                subset = [r for r in results if r.eval_type == et]
                if not subset:
                    continue
                verdicts = [r.verdict for r in subset]
                stats = compute_win_rate(verdicts)
                gen_avg = sum(r.generated_avg_score for r in subset
                              if r.generated_avg_score) / len(subset)
                hum_avg = sum(r.human_avg_score for r in subset
                              if r.human_avg_score) / len(subset)
                wrl = f"{stats['wins']}/{stats['ties']}/{stats['loses']}"
                print(f"{sys_id:<28} {et:>8} {stats['win_rate']:>6.1%} "
                      f"{gen_avg:>6.1f} {hum_avg:>6.1f} {wrl:>10}")

    if args.cwr and all_cwr_results:
        print("\nComparative Win Rate (CWR):")
        print(f"{'Система':<28} {'Тип':>8} {'WR':>6} {'W/T/L':>10}")
        print("-" * 70)
        for sys_id, results in sorted(all_cwr_results.items()):
            for et in eval_types:
                subset = [r for r in results if r.eval_type == et]
                if not subset:
                    continue
                verdicts = [r.verdict for r in subset]
                stats = compute_win_rate(verdicts)
                wrl = f"{stats['wins']}/{stats['ties']}/{stats['loses']}"
                print(f"{sys_id:<28} {et:>8} {stats['win_rate']:>6.1%} {wrl:>10}")

    print(f"\nРезультаты сохранены в: {out_dir}/")
    _save_summary(all_swr_results, all_cwr_results, eval_types, out_dir)


def _save_summary(swr: dict, cwr: dict, eval_types: list[str], out_dir: Path):
    import csv
    rows = []

    for sys_id, results in swr.items():
        for et in eval_types:
            subset = [r for r in results if r.eval_type == et]
            if not subset:
                continue
            verdicts = [r.verdict for r in subset]
            stats = compute_win_rate(verdicts)
            gen_scores = [r.generated_avg_score for r in subset if r.generated_avg_score]
            hum_scores = [r.human_avg_score for r in subset if r.human_avg_score]
            rows.append({
                "system": sys_id, "eval_type": et, "method": "swr",
                "win_rate": f"{stats['win_rate']:.4f}",
                "wins": stats["wins"], "ties": stats["ties"], "loses": stats["loses"],
                "total": stats["total"],
                "generated_avg_score": f"{sum(gen_scores)/len(gen_scores):.2f}" if gen_scores else "",
                "human_avg_score":     f"{sum(hum_scores)/len(hum_scores):.2f}" if hum_scores else "",
            })

    for sys_id, results in cwr.items():
        for et in eval_types:
            subset = [r for r in results if r.eval_type == et]
            if not subset:
                continue
            verdicts = [r.verdict for r in subset]
            stats = compute_win_rate(verdicts)
            rows.append({
                "system": sys_id, "eval_type": et, "method": "cwr",
                "win_rate": f"{stats['win_rate']:.4f}",
                "wins": stats["wins"], "ties": stats["ties"], "loses": stats["loses"],
                "total": stats["total"],
                "generated_avg_score": "", "human_avg_score": "",
            })

    if not rows:
        return
    path = out_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сводная таблица: {path}")


if __name__ == "__main__":
    main()

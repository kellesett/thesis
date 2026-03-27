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
    model: str
    api_key_env: str
    n: int = 1                      # сколько раз вызывать
    _url: Optional[str] = None      # прямой URL (поле url в JSON)
    _url_env: Optional[str] = None  # имя env-переменной с URL (поле url_env в JSON)

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise EnvironmentError(
                f"Переменная {self.api_key_env} не задана в .env (судья '{self.name}')"
            )
        return key

    def get_url(self) -> str:
        if self._url:
            return self._url
        if self._url_env:
            val = os.getenv(self._url_env)
            if not val:
                raise EnvironmentError(
                    f"Переменная {self._url_env} не задана в .env (судья '{self.name}')"
                )
            return val
        raise ValueError(f"Судья '{self.name}': укажи 'url' или 'url_env' в judges.json")

    @property
    def url(self) -> str:
        return self.get_url()

    def make_client(self) -> OpenAI:
        return OpenAI(base_url=self.get_url(), api_key=self.api_key)


def load_judges(path: str) -> list[JudgeConfig]:
    with open(path) as f:
        raw = json.load(f)
    judges = []
    for item in raw:
        judges.append(JudgeConfig(
            name=item["name"],
            model=item["model"],
            api_key_env=item["api_key_env"],
            n=item.get("n", 1),
            _url=item.get("url"),
            _url_env=item.get("url_env"),
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

def load_generated_results(topic: str, results_dir: Path) -> dict[str, dict]:
    """
    Загружает все сгенерированные survey для данной темы из results_dir.
    Ключ — system_id, значение — словарь из JSON-файла.
    """
    results = {}
    if not results_dir.exists():
        return results
    topic_slug = re.sub(r"[^\w]", "_", topic).strip("_")
    for f in results_dir.glob(f"*__{topic_slug}.json"):
        sys_id = f.stem.split("__")[0]
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data.get("generated_text") or data.get("text"):
                results[sys_id] = data
        except Exception:
            pass
    return results


def load_human_surveys(topic: str, k: int = 1) -> list[dict]:
    """
    Загружает до k случайно выбранных human survey для темы.
    Если доступных обзоров меньше k — берёт все.
    Возвращает список dict'ов (может быть пустым).
    """
    import random
    safe = topic.replace("/", "_").replace("\\", "_").replace(" ", "_")
    path = DATASETS_DIR / f"{safe}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    surveys = data.get("surveys", [])
    if not surveys:
        return []
    if k >= len(surveys):
        return surveys
    return random.sample(surveys, k)


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
            gen_text: str, human_texts: list[str],
            eval_type: str, system_id: str) -> SWRPairResult:
    """
    Score Win Rate: оцениваем generated и каждый из human-референсов независимо.
    Скор generated считается один раз, human_avg усредняется по всем k референсам.
    """
    result = SWRPairResult(system_id=system_id, topic=topic, eval_type=eval_type)
    desc = EVAL_TYPE_DESCS[eval_type]

    for judge in judges:
        client = judge.make_client()
        for i in range(judge.n):
            # Оцениваем generated (один раз на судью×прогон)
            prompt_gen = SWR_PROMPT.format(
                topic=topic, eval_type_desc=desc,
                item_label="Generated survey", text=gen_text,
            )
            try:
                resp = call_judge(client, judge.model, prompt_gen)
                result.generated_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response=resp, score=parse_score(resp),
                ))
            except Exception as e:
                result.generated_calls.append(JudgeCall(
                    judge_name=judge.name, judge_model=judge.model,
                    call_index=i, raw_response="", error=str(e),
                ))
            time.sleep(0.5)

            # Оцениваем каждый human-референс, усредняем
            ref_scores = []
            for ref_idx, human_text in enumerate(human_texts):
                prompt_hum = SWR_PROMPT.format(
                    topic=topic, eval_type_desc=desc,
                    item_label=f"Human-written survey (ref {ref_idx + 1}/{len(human_texts)})",
                    text=human_text,
                )
                try:
                    resp = call_judge(client, judge.model, prompt_hum)
                    score = parse_score(resp)
                    ref_scores.append(score)
                    result.human_calls.append(JudgeCall(
                        judge_name=judge.name, judge_model=judge.model,
                        call_index=i * len(human_texts) + ref_idx,
                        raw_response=resp, score=score,
                    ))
                except Exception as e:
                    result.human_calls.append(JudgeCall(
                        judge_name=judge.name, judge_model=judge.model,
                        call_index=i * len(human_texts) + ref_idx,
                        raw_response="", error=str(e),
                    ))
                time.sleep(0.5)

    return result


def run_cwr(judges: list[JudgeConfig], topic: str,
            gen_text: str, human_texts: list[str],
            eval_type: str, system_id: str) -> PairResult:
    """
    Comparative Win Rate: судья сравнивает generated vs каждый human-референс.
    Generated = Survey A, Human = Survey B.
    Победитель определяется большинством голосов по всем парам и всем судьям.
    """
    result = PairResult(system_id=system_id, topic=topic,
                        eval_type=eval_type, method="cwr")
    desc = EVAL_TYPE_DESCS[eval_type]

    for judge in judges:
        client = judge.make_client()
        for i in range(judge.n):
            for ref_idx, human_text in enumerate(human_texts):
                prompt = CWR_PROMPT.format(
                    topic=topic, eval_type_desc=desc,
                    text_a=gen_text, text_b=human_text,
                )
                try:
                    resp = call_judge(client, judge.model, prompt, max_tokens=800)
                    result.judge_calls.append(JudgeCall(
                        judge_name=judge.name, judge_model=judge.model,
                        call_index=i * len(human_texts) + ref_idx,
                        raw_response=resp, winner=parse_winner(resp),
                    ))
                except Exception as e:
                    result.judge_calls.append(JudgeCall(
                        judge_name=judge.name, judge_model=judge.model,
                        call_index=i * len(human_texts) + ref_idx,
                        raw_response="", error=str(e),
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
                        help=f"Папка для eval-результатов (default: {EVAL_OUT_DIR})")
    parser.add_argument("--results-dir", default=None,
                        help="Папка с generated JSON-файлами (default: ROOT/results)")
    parser.add_argument("--resume", action="store_true",
                        help="Пропускать уже посчитанные пары")
    parser.add_argument("--k-refs", type=int, default=1,
                        help="Сколько случайных human-референсов брать на тему (default: 1)")
    args = parser.parse_args()

    # Проверяем что хоть что-то включено
    if not args.swr and not args.cwr:
        parser.error("Укажи хотя бы один метод: --swr и/или --cwr")

    eval_types = [e.strip() for e in args.eval.split(",")]
    for et in eval_types:
        if et not in ("outline", "content"):
            parser.error(f"Неизвестный тип оценки: '{et}'. Допустимо: outline, content")

    # Загружаем судей
    # Резолвим пути относительно ROOT если не абсолютные
    results_dir = Path(args.results_dir) if args.results_dir else ROOT_DIR / "results"
    judges_path = args.judges if Path(args.judges).is_absolute() else ROOT_DIR / args.judges
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
    print(f"Тем: {len(all_topics)}")
    print(f"Human-референсов на тему: {args.k_refs}\n")

    # ── Главный цикл ──────────────────────────────────────────────────────────
    all_swr_results: dict[str, list[SWRPairResult]] = defaultdict(list)
    all_cwr_results: dict[str, list[PairResult]]    = defaultdict(list)

    for topic_data in tqdm(all_topics, desc="Темы"):
        topic = topic_data["topic"]

        # Загружаем k случайных human-референсов
        humans = load_human_surveys(topic, k=args.k_refs)
        if not humans:
            tqdm.write(f"  [SKIP] Нет human survey для '{topic}'. "
                       f"Запусти download_dataset.py")
            continue
        tqdm.write(f"  [::] human refs: {len(humans)} для '{topic[:40]}'")


        # Загружаем сгенерированные результаты
        generated = load_generated_results(topic, results_dir)
        if not generated:
            tqdm.write(f"  [SKIP] Нет сгенерированных результатов для '{topic}'. "
                       f"Запусти base.py")
            continue

        if sys_filter:
            generated = {k: v for k, v in generated.items() if k in sys_filter}

        for sys_id, gen_data in generated.items():
            for eval_type in eval_types:
                gen_text    = get_text_for_eval(gen_data, eval_type)
                human_texts = [get_text_for_eval(h, eval_type) for h in humans]

                safe_topic = re.sub(r'[^\w]', '_', topic)
                for method in methods:
                    out_file = out_dir / f"{sys_id}__{safe_topic}__{eval_type}__{method}.json"

                    if args.resume and out_file.exists():
                        tqdm.write(f"  [SKIP] {sys_id}/{topic[:30]}/{eval_type}/{method}")
                        continue

                    tqdm.write(f"  [{method.upper()}|{eval_type}] {sys_id} / {topic[:45]}")

                    if method == "swr":
                        res = run_swr(judges, topic, gen_text, human_texts,
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
                        res = run_cwr(judges, topic, gen_text, human_texts,
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

# metrics/structural/contradiction/prompts.py
# Статическая часть промпта идёт первой — позволяет провайдеру переиспользовать
# KV-cache для префикса между вызовами. Переменная часть (предложения) — в конце.

TOPIC_FILTER_PROMPT = """Determine whether two sentences from a scientific survey discuss the same specific entity, method, result, or claim.

Two sentences discuss the same subject if they:
- Refer to the same method, model, system, or technique
- Discuss the same experimental result or quantitative claim
- Make assertions about the same dataset, benchmark, or task
- Address the same phenomenon or property of the same object

Two sentences do NOT discuss the same subject if they:
- Are about different methods or different aspects of unrelated systems
- Mention overlapping vocabulary but apply it to distinct objects
- Cover different stages of a pipeline (e.g., training vs inference) without explicit connection

Respond with a JSON object:
{{"same_subject": true or false, "reasoning": "brief explanation"}}

---

Sentence 1: "{s1}"

Sentence 2: "{s2}"
"""

CONTRADICTION_PROMPT = """A contradiction exists when two statements from a scientific survey make incompatible claims about the same entity, method, result, or phenomenon. Note carefully:
- Different formulations of the same fact are NOT contradictions
- Statements about different aspects of the same entity are NOT contradictions
- Statements using different terminology but compatible meanings are NOT contradictions
- Temporal evolution ("earlier work said X, later work showed Y") is NOT a contradiction within the survey's own voice
- Pragmatic incompatibility counts: if one statement implies high cost and another implies accessibility, they contradict
- A genuine contradiction requires that both statements cannot simultaneously be true

Your task: determine whether the two statements below contradict each other.

Respond with a JSON object:
{{"is_contradiction": true or false, "reasoning": "brief explanation", "contradiction_type": "factual" | "methodological" | "quantitative" | "pragmatic" | "none"}}

---

Statement 1 (from section "{section_i}"):
"{s1}"

Statement 2 (from section "{section_j}"):
"{s2}"
"""

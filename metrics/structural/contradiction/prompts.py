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
{schema}

---

Sentence 1: "{s1}"

Sentence 2: "{s2}"
"""

TOPIC_FILTER_SCHEMA_FULL    = '{{"same_subject": true or false, "reasoning": "brief explanation"}}'
TOPIC_FILTER_SCHEMA_COMPACT = '{{"same_subject": true or false}}'

PARAGRAPH_TOPIC_FILTER_PROMPT = """Determine whether two paragraphs from a scientific survey discuss the same specific entity, method, result, benchmark, or claim family closely enough that a contradiction between them is plausible.

Paragraphs discuss the same subject if they:
- Refer to the same method, model, system, dataset, benchmark, or deployment setting
- Make assertions about the same quantitative result, requirement, or capability
- Discuss the same limitation, assumption, or evaluation condition

Paragraphs do NOT discuss the same subject if they:
- Share only broad domain vocabulary
- Discuss different components, methods, or system layers
- Mention related but distinct problems without a shared specific claim

Respond with a JSON object:
{schema}

---

Paragraph 1: "{s1}"

Paragraph 2: "{s2}"
"""

CONTRADICTION_PROMPT = """A contradiction exists only when two statements from a scientific survey make clearly incompatible claims about the same specific entity, method, result, or phenomenon under the same scope, time, and conditions. Be conservative.

Mark a contradiction only if both statements cannot simultaneously be true.

Do NOT mark as contradiction when the difference can be explained by:
- Different formulations of the same fact
- Statements about different aspects of the same entity
- A broad category vs a specific instance or subtype
- A general capability vs a limitation for common/default applications
- An optional/possible deployment mode vs a chosen/default/preferred deployment mode
- A conditional claim vs an unconditional claim, unless the condition is explicitly satisfied
- A system being under test, partially deployed, or discussed at different maturity stages
- Temporal evolution ("earlier work said X, later work showed Y") within the survey's own voice
- A limitation, caveat, or implementation detail that narrows but does not negate another claim
- Pragmatic tension or inconvenience, unless one statement clearly makes the other impossible

Your task: determine whether the two statements below contradict each other.

Respond with a JSON object:
{schema}

---

Statement 1 (from section "{section_i}"):
"{s1}"

Statement 2 (from section "{section_j}"):
"{s2}"
"""

CONTRADICTION_SCHEMA_FULL    = '{{"is_contradiction": true or false, "reasoning": "brief explanation", "contradiction_type": "factual" | "methodological" | "quantitative" | "pragmatic" | "none"}}'
CONTRADICTION_SCHEMA_COMPACT = '{{"is_contradiction": true or false, "contradiction_type": "factual" | "methodological" | "quantitative" | "pragmatic" | "none"}}'

PARAGRAPH_CONTRADICTION_PROMPT = """You will compare two paragraphs from a scientific survey.

Find whether any statement in Paragraph 1 clearly contradicts any statement in Paragraph 2.

Use the same conservative definition as for sentence-level checks:
- A contradiction requires incompatible claims about the same specific entity, method, result, or phenomenon under the same scope, time, and conditions.
- Do NOT mark broad-vs-specific differences, caveats, limitations, optional-vs-default implementation details, temporal evolution, or pragmatic tension as contradictions unless one statement clearly makes the other impossible.

If a contradiction exists, return the minimal statement excerpts that contradict each other. Prefer exact sentence text from the paragraphs when possible. If the paragraph contains sentence IDs such as "[synthetic_01.S001]", preserve those IDs in the returned excerpts.

Respond with a JSON object:
{schema}

---

Paragraph 1 (from section "{section_i}"):
"{s1}"

Paragraph 2 (from section "{section_j}"):
"{s2}"
"""

PARAGRAPH_CONTRADICTION_SCHEMA_FULL = '{{"has_contradiction": true or false, "contradictions": [{{"statement_1": "minimal exact excerpt from paragraph 1", "statement_2": "minimal exact excerpt from paragraph 2", "contradiction_type": "factual" | "methodological" | "quantitative" | "pragmatic", "reasoning": "brief explanation"}}]}}'
PARAGRAPH_CONTRADICTION_SCHEMA_COMPACT = '{{"has_contradiction": true or false, "contradictions": [{{"statement_1": "minimal exact excerpt from paragraph 1", "statement_2": "minimal exact excerpt from paragraph 2", "contradiction_type": "factual" | "methodological" | "quantitative" | "pragmatic"}}]}}'

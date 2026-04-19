"""Shared classifier prompts for the factuality metric.

Single source of truth — ``main.py`` and ``validate.py`` both import from
here instead of keeping their own copies. Prior to consolidation the strings
lived inline in both modules with a "copied from main.py" comment, which is
drift-prone.

Prompt structure is deliberately cache-friendly:
    * system message = ALL static content (instructions + category
      definitions + response schema).
    * user message   = ONLY the variable content ({claim}, {source_context}).

Providers that support prefix caching (Anthropic, some Google Gemini) can
reuse the ~500-token system prefix across calls; the variable user payload
stays small. With non-caching providers behaviour is unchanged.
"""
from __future__ import annotations


# The system prompt carries every token that's identical across calls —
# instructions, the fixed category schema, and the output JSON contract. Do
# NOT inject per-call variables here; that would break caching.
CATEGORY_SYSTEM = """\
You are classifying atomic claims from a scientific survey into one of four \
categories based on what type of information they convey.

Categories:
A — General topical claims. Statements that can be made based on only \
reading the abstract. They describe what a paper is about, what problem it \
addresses, or its general contribution without specific methodological or \
quantitative details.
B — Methodological refinements. Statements containing specific details \
about how methods work, requiring information typically found in the \
Methods section.
C — Quantitative claims. Statements containing specific numerical values, \
typically from the Results section.
D — Critical or comparative claims. Statements making evaluative judgments, \
comparing approaches, discussing limitations, or pointing to contradictions \
— typically from Discussion or Related Work sections.

Respond with a JSON object and nothing else:
{"category": "A" | "B" | "C" | "D", "reasoning": "brief explanation", \
"confidence": "high" | "medium" | "low"}"""


# The user prompt carries ONLY what changes between calls. Format with
# ``.format(claim=..., source_context=...)``. Sized limits (600/800 chars)
# are applied by the caller before formatting.
CATEGORY_PROMPT = """\
Claim:
"{claim}"

Source paper context (if available):
"{source_context}"
"""

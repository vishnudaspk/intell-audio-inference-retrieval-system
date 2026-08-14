"""
Query Understanding Service - Phase 7B.

Extracts intent, actions, objects, targets, and content-type preferences from a
natural-language query using a deterministic rule-based pipeline. An optional
LLM fallback handles queries that match no rules.

The rule-based path never calls the LLM, is fully tested, and always returns a
QueryIntent even on failure.
"""

import re
from typing import List, Optional, Tuple

from schemas.models import QueryIntent
from utils.logger import logger

# Intent rule table
# Each entry: (pattern, intent_label, content_type_preferences)
_INTENT_RULES: List[Tuple[str, str, List[str]]] = [
    # Procedural / instructional
    (r"\bhow\s+(to|do\s+i|can\s+i|do\s+you)\b", "procedural_instruction", ["instruction", "procedure"]),
    (r"\bwhat\s+steps?\b", "procedure_query", ["procedure", "instruction"]),
    (r"\bstep[s]?\s+to\b", "procedure_query", ["procedure", "instruction"]),
    (r"\bprocedure\s+(for|to)\b", "procedure_query", ["procedure", "instruction"]),
    # Warning / danger
    (r"\bwhat\s+(warning|danger|caution|risk)\b", "warning", ["warning"]),
    (r"\bsafe\s+to\b", "warning", ["warning"]),
    # Comparison
    (r"\bdifference\s+between\b", "comparison", ["comparison"]),
    (r"\bcompare\b", "comparison", ["comparison"]),
    (r"\bvs\.?\b", "comparison", ["comparison"]),
    # Definition
    (r"\bwhat\s+is\s+a?\b", "definition", ["definition", "explanation"]),
    (r"\bdefine\b", "definition", ["definition"]),
    (r"\bmeaning\s+of\b", "definition", ["definition"]),
    # Explanation / location
    (r"\bwhere\s+does\s+(he|she|it|the\s+\w+)\s+(explain|mention|talk|discuss)\b", "procedural_instruction", ["explanation", "instruction"]),
    (r"\bwhere\s+(is|are)\b", "temporal_location", []),
    (r"\bwhen\b", "temporal_location", []),
    (r"\bat\s+what\s+point\b", "temporal_location", []),
    # Speaker query
    (r"\bwho\b", "speaker_query", []),
    (r"\bwhich\s+speaker\b", "speaker_query", []),
    # Troubleshooting
    (r"\b(fix|repair|troubleshoot|debug|not\s+working)\b", "troubleshooting", ["troubleshooting"]),
    (r"\bwhy\s+(is|does|did|isn.t|doesn.t)\b", "troubleshooting", ["troubleshooting", "explanation"]),
    # Recommendation
    (r"\bshould\s+i\b", "recommendation", ["recommendation"]),
    (r"\bwhich\s+(one|option|bolt|part|tool)\b", "recommendation", ["recommendation", "instruction"]),
]

# Verbs that signal action extraction
_ACTION_VERBS = {
    "remove", "unscrew", "install", "replace", "attach", "detach", "connect",
    "disconnect", "tighten", "loosen", "cut", "drill", "lift", "push", "pull",
    "turn", "adjust", "set", "reset", "align", "check", "inspect", "clean",
    "lubricate", "seal", "open", "close", "start", "stop", "test", "fix",
    "repair", "mount", "unmount", "plug", "unplug", "press",
}

# Preposition patterns for target extraction
_TARGET_PATTERNS = [
    r"(?:to|from|off|of)\s+(?:the|a|an)?\s*([a-z][a-z\s\-]{1,30}?)(?:\s*[,\.?]|$)",
]

# Tool markers
_TOOL_MARKERS = [
    r"(?:using|with)\s+(?:a|an|the)?\s*([a-z][a-z0-9\s\-]{1,30}?)(?:\s*[,\.]|$)",
]


class QueryUnderstanding:
    """
    Rule-based query understanding service.
    Extracts QueryIntent from natural-language queries without LLM by default.
    Optionally falls back to LLM for unknown intent when llm_provider is supplied.
    """

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    def extract(self, query: str) -> QueryIntent:
        """Extract QueryIntent from a query string. Always returns a QueryIntent, never raises."""
        try:
            return self._extract_impl(query)
        except Exception as exc:
            logger.warning(f"QueryUnderstanding.extract failed, returning unknown intent: {exc}")
            return QueryIntent(
                query=query,
                normalized_query=query.strip().lower(),
                intent="unknown",
            )

    def _extract_impl(self, query: str) -> QueryIntent:
        normalized = query.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)

        # 1. Intent detection via rule table
        intent = "unknown"
        content_type_prefs: List[str] = []
        for pattern, rule_intent, rule_prefs in _INTENT_RULES:
            if re.search(pattern, normalized):
                intent = rule_intent
                content_type_prefs = rule_prefs
                break

        # 2. Action extraction
        tokens = re.sub(r"[^\w\s]", " ", normalized).split()
        actions: List[str] = [t for t in tokens if t in _ACTION_VERBS]

        # 3. Object extraction:
        # a) Nouns following question words like "which <object>", "what <object>"
        # b) Nouns following action verbs
        objects: List[str] = []
        for i, tok in enumerate(tokens):
            # Question word pattern: "which bolt", "what bolt"
            if tok in ("which", "what") and i + 1 < len(tokens):
                cand = tokens[i + 1]
                if cand not in _ACTION_VERBS and cand not in ("is", "are", "do", "does", "did", "to", "the", "a", "an", "one") and len(cand) > 2:
                    objects.append(cand)
            # Action verb pattern: "remove bolt", "remove the bolt"
            if tok in _ACTION_VERBS and i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if next_tok in ("a", "an", "the") and i + 2 < len(tokens):
                    next_tok = tokens[i + 2]
                if next_tok not in _ACTION_VERBS and len(next_tok) > 2:
                    objects.append(next_tok)

        # 4. Target extraction: "to/from the [TARGET]" patterns
        targets: List[str] = []
        for pat in _TARGET_PATTERNS:
            for m in re.finditer(pat, normalized):
                candidate = m.group(1).strip()
                if candidate and len(candidate) > 2:
                    targets.append(candidate)

        # 5. Tool extraction: "using/with a [TOOL]" patterns
        tools: List[str] = []
        for pat in _TOOL_MARKERS:
            for m in re.finditer(pat, normalized):
                candidate = m.group(1).strip()
                if candidate and len(candidate) > 2:
                    tools.append(candidate)

        # 6. Entity extraction: capitalized nouns from original query
        entities: List[str] = []
        for tok in query.split():
            clean = re.sub(r"[^\w]", "", tok)
            if clean and clean[0].isupper() and len(clean) > 2:
                entities.append(clean.lower())

        # 7. Decide if LLM fallback needed
        requires_llm = intent == "unknown" and not actions

        # 8. Optional LLM fallback
        if requires_llm and self._llm is not None:
            try:
                llm_intent = self._llm_extract(query, normalized)
                if llm_intent:
                    intent = llm_intent.get("intent", intent)
                    actions = list(set(actions + llm_intent.get("actions", [])))
                    objects = list(set(objects + llm_intent.get("objects", [])))
                    targets = list(set(targets + llm_intent.get("targets", [])))
                    tools = list(set(tools + llm_intent.get("tools", [])))
                    content_type_prefs = llm_intent.get("content_type_preferences", content_type_prefs)
                    requires_llm = False
            except Exception as llm_exc:
                logger.warning(f"LLM query understanding fallback failed: {llm_exc}.")

        return QueryIntent(
            query=query,
            normalized_query=normalized,
            intent=intent,
            actions=actions,
            objects=objects,
            targets=targets,
            entities=entities,
            tools=tools,
            content_type_preferences=content_type_prefs,
            requires_llm=requires_llm,
        )

    def _llm_extract(self, query: str, normalized: str) -> Optional[dict]:
        """Call LLM for structured intent extraction. Returns dict or None."""
        import json

        prompt = (
            f"Extract query intent from the user question below.\n"
            f"Return valid JSON with keys: intent (string), actions (list), objects (list), "
            f"targets (list), tools (list), content_type_preferences (list).\n\n"
            f"Question: {query}\n"
        )
        system_prompt = (
            "You are a query understanding assistant. "
            "Return only valid JSON. Do not include explanation or markdown."
        )
        try:
            raw = self._llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                json_mode=True,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(lines[1:-1]).strip()
            return json.loads(cleaned)
        except Exception:
            return None

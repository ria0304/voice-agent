"""
Intent classifier.

Supports compound commands, e.g.:
    "Summarize this text and save it to summary.txt"
→ intents: ["summarize_text", "create_file"]

Supported intents
-----------------
create_file    – create an empty file or folder
write_code     – generate code and write to file
summarize_text – summarise provided text
general_chat   – everything else
"""

import json
import re
from agent.llm import get_llm_response

INTENT_SYSTEM = """You are an intent classifier for a voice-controlled local AI agent.

Given a user's command, identify ALL applicable intents and extract relevant parameters.

## Supported Intents
- create_file      : User wants to create a file or folder
- write_code       : User wants to generate code and save it to a file
- summarize_text   : User wants text to be summarised
- general_chat     : General question, conversation, or anything not covered above

## Notes
- A single command can trigger MULTIPLE intents (compound commands), e.g.
  "Write a Python sort function and save it to sort.py" → ["write_code", "create_file"]
  "Summarize this and save to summary.txt"              → ["summarize_text", "create_file"]
- For write_code, always include create_file as well if a filename is mentioned.

## Output format
Respond ONLY with valid JSON (no markdown fences):
{
  "intents": ["intent1", "intent2"],
  "params": {
    "filename": "optional_filename.ext or null",
    "language": "python/javascript/… or null",
    "text_to_summarize": "text content if present, else null",
    "description": "short description of what the user wants"
  }
}
"""


def classify_intent(text: str, config: dict = None) -> dict:
    """
    Classify the intent(s) of the transcribed text.

    Returns
    -------
    dict with keys:
        intents : list[str]
        params  : dict
    """
    if not text or not text.strip():
        return {"intents": ["general_chat"], "params": {"description": "empty input"}}

    # Fast keyword pre-filter (optional fallback if LLM unavailable)
    try:
        raw = get_llm_response(
            prompt=text,
            system=INTENT_SYSTEM,
            config=config,
            json_mode=True,
        )
        result = _parse_json(raw)
        # Validate and sanitise
        valid_intents = {"create_file", "write_code", "summarize_text", "general_chat"}
        result["intents"] = [
            i for i in result.get("intents", ["general_chat"]) if i in valid_intents
        ] or ["general_chat"]
        if "params" not in result:
            result["params"] = {}
        return result

    except Exception:
        # Graceful degradation: keyword fallback
        return _keyword_fallback(text)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    return json.loads(clean)


def _keyword_fallback(text: str) -> dict:
    """Simple keyword-based fallback when LLM is unavailable."""
    lower = text.lower()
    intents = []
    params = {}

    code_words = ["write code", "generate code", "create a function", "write a function",
                  "write a script", "write a class", "code that"]
    file_words = ["create a file", "make a file", "save to", "write to", "new file"]
    sum_words  = ["summarize", "summarise", "summary", "tldr", "brief"]

    if any(w in lower for w in code_words):
        intents.append("write_code")
    if any(w in lower for w in file_words):
        intents.append("create_file")
    if any(w in lower for w in sum_words):
        intents.append("summarize_text")

    # Remove duplicates, keep order
    seen = set()
    unique = []
    for i in intents:
        if i not in seen:
            seen.add(i)
            unique.append(i)

    if not unique:
        unique = ["general_chat"]

    # Try to extract filename
    fn_match = re.search(r'\b([\w\-]+\.[a-zA-Z]{1,5})\b', text)
    if fn_match:
        params["filename"] = fn_match.group(1)

    params["description"] = text[:100]
    return {"intents": unique, "params": params}

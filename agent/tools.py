

import os
import re
import json
import datetime
from pathlib import Path
from agent.llm import get_llm_response

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def execute_tool(
    intents: list,
    params: dict,
    transcribed_text: str,
    config: dict = None,
    chat_context: list = None,
) -> dict:
    """
    Route to the correct tool(s) and return a result dict.

    Returns
    -------
    dict:
        success : bool
        action  : str  (human-readable description of what happened)
        output  : str  (content to display in the UI)
        files   : list[str]  (paths of files created/modified)
    """
    chat_context = chat_context or []
    primary = intents[0] if intents else "general_chat"

    # ── Compound: summarize + create_file ─────────────────────────────────────
    if "summarize_text" in intents and "create_file" in intents:
        summary = _summarize(transcribed_text, params, config)
        filename = params.get("filename") or _make_filename("summary", ".txt")
        path = _safe_path(filename)
        path.write_text(summary, encoding="utf-8")
        return {
            "success": True,
            "action": f"Summarized text → saved to output/{path.name}",
            "output": summary,
            "files": [str(path)],
        }

    # ── Compound: write_code + create_file ───────────────────────────────────
    if "write_code" in intents:
        return _write_code(transcribed_text, params, config)

    # ── Single intent routing ─────────────────────────────────────────────────
    if primary == "create_file":
        return _create_file(transcribed_text, params)

    if primary == "summarize_text":
        summary = _summarize(transcribed_text, params, config)
        return {
            "success": True,
            "action": "Summarized text",
            "output": summary,
            "files": [],
        }

    if primary == "general_chat":
        return _general_chat(transcribed_text, params, config, chat_context)

    return {
        "success": False,
        "action": f"Unhandled intent: {primary}",
        "output": "Sorry, I don't know how to handle that yet.",
        "files": [],
    }


# ── Tools ──────────────────────────────────────────────────────────────────────

def _write_code(text: str, params: dict, config: dict) -> dict:
    """Generate code with LLM and write it to a file."""
    lang = params.get("language") or _infer_language(text)
    ext_map = {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "java": ".java", "c": ".c", "cpp": ".cpp", "go": ".go",
        "rust": ".rs", "bash": ".sh", "html": ".html", "css": ".css",
        "sql": ".sql", "json": ".json", "yaml": ".yaml",
    }
    ext = ext_map.get(lang.lower(), ".txt") if lang else ".py"
    filename = params.get("filename") or _make_filename("code", ext)
    path = _safe_path(filename)

    system = f"""You are an expert {lang or 'software'} developer.
Generate clean, well-commented, production-quality code.
Respond ONLY with the raw code — no markdown fences, no explanation outside comments."""

    code = get_llm_response(
        prompt=f"Write the following: {text}",
        system=system,
        config=config,
    )

    # Strip accidental fences
    code = _strip_fences(code)
    path.write_text(code, encoding="utf-8")

    return {
        "success": True,
        "action": f"Generated {lang or 'code'} → saved to output/{path.name}",
        "output": code,
        "files": [str(path)],
    }


def _create_file(text: str, params: dict) -> dict:
    """Create an empty file or folder."""
    filename = params.get("filename")
    if not filename:
        # Try to extract from natural language
        m = re.search(r'(?:called?|named?|file)\s+([\w\-\.]+)', text, re.I)
        filename = m.group(1) if m else _make_filename("new_file", ".txt")

    path = _safe_path(filename)

    if path.suffix == "" or filename.endswith("/"):
        # It's a directory
        path.mkdir(parents=True, exist_ok=True)
        return {
            "success": True,
            "action": f"Created folder: output/{path.name}/",
            "output": f"📁 Directory created: output/{path.name}/",
            "files": [str(path)],
        }
    else:
        path.touch()
        return {
            "success": True,
            "action": f"Created file: output/{path.name}",
            "output": f"📄 Empty file created: output/{path.name}",
            "files": [str(path)],
        }


def _summarize(text: str, params: dict, config: dict) -> str:
    """Summarize the text in the command (or the explicit text_to_summarize param)."""
    content = params.get("text_to_summarize") or text

    # If text is just the command (short), ask LLM to summarize what it can
    system = """You are a precise summarizer.
Produce a clear, concise summary in plain prose.
Keep it under 150 words unless the source is very long.
Do not add preamble like 'Here is a summary:'."""

    return get_llm_response(
        prompt=f"Summarize the following:\n\n{content}",
        system=system,
        config=config,
    )


def _general_chat(text: str, params: dict, config: dict, chat_context: list) -> dict:
    """Handle general conversation with chat memory."""
    system = """You are a helpful, concise voice AI assistant.
Respond naturally and conversationally. Keep answers brief unless a detailed explanation is needed."""

    # Build messages with context
    messages = [{"role": "system", "content": system}]
    messages.extend(chat_context[-10:])  # last 5 turns
    messages.append({"role": "user", "content": text})

    response = get_llm_response(
        prompt=text,
        system=system,
        config=config,
        messages=messages,
    )

    return {
        "success": True,
        "action": "General chat response",
        "output": response,
        "files": [],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_path(filename: str) -> Path:
    """Ensure the path stays inside the output/ directory."""
    # Strip any path traversal attempts
    safe_name = Path(filename).name
    return OUTPUT_DIR / safe_name


def _make_filename(prefix: str, ext: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{ext}"


def _infer_language(text: str) -> str:
    lower = text.lower()
    langs = {
        "python": ["python", ".py", "django", "flask", "pandas", "numpy"],
        "javascript": ["javascript", "js", "node", "react", "vue"],
        "typescript": ["typescript", "ts"],
        "bash": ["bash", "shell", "script"],
        "java": ["java", "spring"],
        "go": [" go ", "golang"],
        "rust": ["rust"],
        "sql": ["sql", "query", "database"],
        "html": ["html", "webpage", "web page"],
    }
    for lang, keywords in langs.items():
        if any(k in lower for k in keywords):
            return lang
    return "python"  # default


def _strip_fences(code: str) -> str:
    """Remove markdown code fences if the LLM added them."""
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code.strip()

import streamlit as st
import os
import json
import time
from pathlib import Path
from agent.stt import transcribe_audio
from agent.intent_classifier import classify_intent
from agent.tools import execute_tool
from agent.llm import get_llm_response

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceAgent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #7c6af7;
    --accent2: #f76ac8;
    --text: #e8e8f0;
    --text-muted: #6a6a8a;
    --success: #4af7a0;
    --warning: #f7c44a;
    --error: #f76a6a;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3 { font-family: var(--mono) !important; }

.main-header {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -1px;
}

.sub-header {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-family: var(--mono);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.pipeline-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.6rem 0;
    transition: border-color 0.3s;
}

.pipeline-card:hover { border-color: var(--accent); }

.pipeline-card .label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}

.pipeline-card .value {
    font-size: 1rem;
    color: var(--text);
    word-break: break-word;
}

.intent-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.intent-create_file    { background: rgba(124,106,247,0.2); color: #7c6af7; border: 1px solid #7c6af7; }
.intent-write_code     { background: rgba(74,247,160,0.2); color: #4af7a0; border: 1px solid #4af7a0; }
.intent-summarize_text { background: rgba(247,196,74,0.2); color: #f7c44a; border: 1px solid #f7c44a; }
.intent-general_chat   { background: rgba(247,106,200,0.2); color: #f76ac8; border: 1px solid #f76ac8; }

.output-block {
    background: #0d0d14;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--success);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
}

.status-ok   { color: var(--success); }
.status-err  { color: var(--error); }

.history-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}

.step-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-muted);
}

.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

div[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_context" not in st.session_state:
    st.session_state.chat_context = []

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    stt_provider = st.selectbox(
        "STT Provider",
        ["whisper-local", "groq", "openai"],
        help="Speech-to-text engine"
    )
    
    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "anthropic", "openai"],
        help="Language model backend"
    )
    
    if llm_provider == "ollama":
        ollama_model = st.text_input("Ollama Model", value="llama3", placeholder="llama3, mistral …")
    
    st.markdown("---")
    st.markdown("**Output directory**")
    st.code("./output/", language="bash")
    
    output_files = list(Path("output").glob("*")) if Path("output").exists() else []
    if output_files:
        st.markdown(f"**{len(output_files)} file(s) created:**")
        for f in sorted(output_files)[-5:]:
            st.markdown(f"• `{f.name}`")
    
    st.markdown("---")
    st.markdown("**Session history**")
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.chat_context = []
        st.rerun()
    st.markdown(f"Actions taken: **{len(st.session_state.history)}**")

# ── Main layout ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎙️ VoiceAgent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Local AI • Speech → Intent → Action</div>', unsafe_allow_html=True)

input_col, pipeline_col = st.columns([1, 1], gap="large")

# ── Input column ────────────────────────────────────────────────────────────────
with input_col:
    st.markdown("### Input")
    
    input_mode = st.radio("Mode", ["Upload audio file", "Type text directly"], horizontal=True, label_visibility="collapsed")
    
    audio_file = None
    text_input = ""
    
    if input_mode == "Upload audio file":
        audio_file = st.file_uploader(
            "Drop an audio file here",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="WAV, MP3, M4A, OGG, FLAC supported"
        )
        if audio_file:
            st.audio(audio_file)
    else:
        text_input = st.text_area(
            "Type your command",
            placeholder='e.g. "Create a Python file with a retry function" or "Summarize: The quick brown fox…"',
            height=100
        )
    
    # Confirmation gate for file ops (Human-in-the-Loop bonus)
    confirm_before_exec = st.checkbox("⚠️ Confirm before executing file operations", value=True)
    
    run_btn = st.button("▶ RUN AGENT", use_container_width=True)

# ── Pipeline column ─────────────────────────────────────────────────────────────
with pipeline_col:
    st.markdown("### Pipeline")
    
    transcription_placeholder = st.empty()
    intent_placeholder = st.empty()
    action_placeholder = st.empty()
    output_placeholder = st.empty()

    def render_pipeline_card(label, value, extra_html=""):
        return f"""
        <div class="pipeline-card">
            <div class="label">{label}</div>
            <div class="value">{value}{extra_html}</div>
        </div>
        """

    # Initial placeholders
    transcription_placeholder.markdown(render_pipeline_card("01 — Transcription", "<span style='color:#3a3a55'>Awaiting audio…</span>"), unsafe_allow_html=True)
    intent_placeholder.markdown(render_pipeline_card("02 — Intent", "<span style='color:#3a3a55'>—</span>"), unsafe_allow_html=True)
    action_placeholder.markdown(render_pipeline_card("03 — Action", "<span style='color:#3a3a55'>—</span>"), unsafe_allow_html=True)
    output_placeholder.markdown(render_pipeline_card("04 — Output", "<span style='color:#3a3a55'>—</span>"), unsafe_allow_html=True)

# ── Execution ───────────────────────────────────────────────────────────────────
if run_btn:
    config = {
        "stt_provider": stt_provider,
        "llm_provider": llm_provider,
        "ollama_model": ollama_model if llm_provider == "ollama" else None,
    }
    
    with st.spinner(""):
        # ── Step 1: STT ──
        if audio_file:
            transcription_placeholder.markdown(
                render_pipeline_card("01 — Transcription", "<span style='color:#f7c44a'>Transcribing…</span>"),
                unsafe_allow_html=True
            )
            try:
                transcribed_text = transcribe_audio(audio_file, provider=stt_provider)
            except Exception as e:
                transcribed_text = None
                st.error(f"STT failed: {e}")
        elif text_input.strip():
            transcribed_text = text_input.strip()
        else:
            st.warning("Please provide audio or type a command.")
            st.stop()
        
        if not transcribed_text:
            st.error("Could not transcribe audio. Check your STT provider settings.")
            st.stop()
        
        transcription_placeholder.markdown(
            render_pipeline_card("01 — Transcription", transcribed_text),
            unsafe_allow_html=True
        )
        time.sleep(0.2)
        
        # ── Step 2: Intent ──
        intent_placeholder.markdown(
            render_pipeline_card("02 — Intent", "<span style='color:#f7c44a'>Classifying…</span>"),
            unsafe_allow_html=True
        )
        try:
            intent_result = classify_intent(transcribed_text, config=config)
        except Exception as e:
            st.error(f"Intent classification failed: {e}")
            st.stop()
        
        intents = intent_result.get("intents", ["general_chat"])
        primary_intent = intents[0] if intents else "general_chat"
        params = intent_result.get("params", {})
        
        badge_html = " ".join([
            f'<span class="intent-badge intent-{i}">{i.replace("_"," ")}</span>'
            for i in intents
        ])
        intent_placeholder.markdown(
            render_pipeline_card("02 — Intent", badge_html),
            unsafe_allow_html=True
        )
        time.sleep(0.2)
        
        # ── Step 3: Confirmation gate ──
        needs_file_op = any(i in ["create_file", "write_code"] for i in intents)
        confirmed = True
        
        if confirm_before_exec and needs_file_op:
            action_placeholder.markdown(
                render_pipeline_card("03 — Action", "<span style='color:#f7c44a'>⚠️ Confirm in sidebar →</span>"),
                unsafe_allow_html=True
            )
            with st.sidebar:
                st.warning(f"⚠️ File operation requested: **{primary_intent}**")
                col1, col2 = st.columns(2)
                confirmed = col1.button("✅ Confirm", key="confirm_exec")
                cancelled = col2.button("❌ Cancel", key="cancel_exec")
                if cancelled:
                    st.info("Operation cancelled.")
                    st.stop()
                if not confirmed:
                    st.info("Waiting for confirmation…")
                    st.stop()
        
        # ── Step 4: Execute ──
        action_placeholder.markdown(
            render_pipeline_card("03 — Action", "<span style='color:#f7c44a'>Executing…</span>"),
            unsafe_allow_html=True
        )
        
        try:
            result = execute_tool(
                intents=intents,
                params=params,
                transcribed_text=transcribed_text,
                config=config,
                chat_context=st.session_state.chat_context,
            )
        except Exception as e:
            result = {"success": False, "action": "error", "output": str(e)}
        
        # Update chat context for general_chat
        if primary_intent == "general_chat":
            st.session_state.chat_context.append({"role": "user", "content": transcribed_text})
            st.session_state.chat_context.append({"role": "assistant", "content": result.get("output", "")})
            if len(st.session_state.chat_context) > 20:
                st.session_state.chat_context = st.session_state.chat_context[-20:]
        
        action_str = result.get("action", "—")
        status_class = "status-ok" if result.get("success") else "status-err"
        status_icon = "✓" if result.get("success") else "✗"
        
        action_placeholder.markdown(
            render_pipeline_card(
                "03 — Action",
                f'<span class="{status_class}">{status_icon} {action_str}</span>'
            ),
            unsafe_allow_html=True
        )
        
        output_text = result.get("output", "No output.")
        output_placeholder.markdown(
            render_pipeline_card(
                "04 — Output",
                f'<div class="output-block">{output_text}</div>'
            ),
            unsafe_allow_html=True
        )
        
        # ── Save to history ──
        st.session_state.history.append({
            "transcript": transcribed_text,
            "intents": intents,
            "action": action_str,
            "output_snippet": output_text[:120] + ("…" if len(output_text) > 120 else ""),
            "success": result.get("success", False),
        })

# ── History panel ───────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Session History")
    
    for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
        icon = "✅" if item["success"] else "❌"
        badges = " ".join([f'`{intent}`' for intent in item["intents"]])
        with st.expander(f"{icon} [{len(st.session_state.history)-i+1}] {item['transcript'][:60]}…"):
            st.markdown(f"**Intent(s):** {badges}")
            st.markdown(f"**Action:** `{item['action']}`")
            st.markdown(f"**Output:** {item['output_snippet']}")

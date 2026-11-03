
import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from coach import (
    CoachClient,
    build_system_prompt,
    parse_coach_response,
)

from speech import (
    get_ollama_models,
    transcribe_parakeet,
    synthesize_speech,
)

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_DIR = Path(__file__).parent
SESSION_DIR = APP_DIR / "data" / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Local Speaking Trainer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# TRANSLATIONS
# ============================================================

LABELS = {
    "English": {
        "title": "Local Speaking Trainer",
        "subtitle": "Your private AI English and German speaking coach.",
        "settings": "Practice Settings",
        "language": "Practice Language",
        "interface": "Interface Language",
        "level": "Your Language Level",
        "focus": "Learning Focus",
        "scenario": "Conversation Scenario",
        "correction": "Correction Intensity",
        "feedback": "Feedback Language",
        "goal": "Personal Learning Goal",
        "length": "Coach Response Length",
        "model": "Ollama Model",
        "url": "Ollama API URL",
        "refresh": "Refresh Installed Models",
        "voice": "Coach Voice",
        "tts_url": "TTS Studio URL",
        "tts_model": "TTS Model",
        "token": "TTS API Token",
        "asr": "Parakeet Model",
        "threads": "CPU Threads",
        "temperature": "LLM Temperature",
        "conversation": "Conversation",
        "record": "Record Your Answer",
        "record_help": "Record one answer, then submit it.",
        "send_voice": "Send Voice Answer",
        "type": "Or Type Your Answer",
        "send_text": "Send Text Answer",
        "thinking": "Your coach is thinking...",
        "transcribing": "Parakeet is transcribing...",
        "speaking": "Generating coach voice...",
        "you": "You",
        "coach": "Coach",
        "correction": "Correction",
        "explanation": "Explanation",
        "vocabulary": "Useful Vocabulary",
        "repeat": "Repeat Coach Response",
        "simplify": "Explain More Simply",
        "clear": "New Conversation",
        "turns": "Turns",
        "words": "Your Words",
        "corrections": "Corrections",
        "save": "Save Session Locally",
        "show_transcript": "Show Recognized Transcript",
        "empty": "Start by speaking or typing in English or German.",
        "autoplay": "If your browser blocks automatic playback, use the audio player.",
        "missing_token": "Configure your TTS API token to enable spoken replies.",
        "diagnostics": "Diagnostics",
        "stop": "Record an answer. Your coach will respond automatically.",
        "no_speech": "No speech detected. Please try again.",
        "reset": "Start a Fresh Practice Session",
        "history": "Session History",
        "no_models": "Could not retrieve installed Ollama models.",
        "tts_error": "TTS generation failed",
        "ollama_error": "Ollama request failed",
        "asr_error": "Parakeet transcription failed",
        "session_saved": "Session saved.",
        "download": "Download Session",
        "listening": "Ready for your next answer.",
    },

    "Deutsch": {
        "title": "Lokaler Sprachtrainer",
        "subtitle": "Dein privater KI-Sprachcoach für Englisch und Deutsch.",
        "settings": "Übungseinstellungen",
        "language": "Übungssprache",
        "interface": "Oberflächensprache",
        "level": "Dein Sprachniveau",
        "focus": "Lernschwerpunkt",
        "scenario": "Gesprächssituation",
        "correction": "Korrekturintensität",
        "feedback": "Sprache der Erklärungen",
        "goal": "Persönliches Lernziel",
        "length": "Antwortlänge des Coachs",
        "model": "Ollama-Modell",
        "url": "Ollama API URL",
        "refresh": "Installierte Modelle aktualisieren",
        "voice": "Coach-Stimme",
        "tts_url": "TTS Studio URL",
        "tts_model": "TTS-Modell",
        "token": "TTS API-Token",
        "asr": "Parakeet-Modell",
        "threads": "CPU-Threads",
        "temperature": "LLM-Temperatur",
        "conversation": "Gespräch",
        "record": "Antwort aufnehmen",
        "record_help": "Nimm eine Antwort auf und sende sie ab.",
        "send_voice": "Sprachantwort senden",
        "type": "Oder Antwort eintippen",
        "send_text": "Textantwort senden",
        "thinking": "Dein Coach denkt nach...",
        "transcribing": "Parakeet transkribiert...",
        "speaking": "Coach-Stimme wird erzeugt...",
        "you": "Du",
        "coach": "Coach",
        "correction": "Korrektur",
        "explanation": "Erklärung",
        "vocabulary": "Nützlicher Wortschatz",
        "repeat": "Coach-Antwort wiederholen",
        "simplify": "Einfacher erklären",
        "clear": "Neues Gespräch",
        "turns": "Gesprächsrunden",
        "words": "Deine Wörter",
        "corrections": "Korrekturen",
        "save": "Gespräch lokal speichern",
        "show_transcript": "Erkannten Text anzeigen",
        "empty": "Beginne mit einer Sprachaufnahme oder Texteingabe.",
        "autoplay": "Falls dein Browser die automatische Wiedergabe blockiert, nutze den Audioplayer.",
        "missing_token": "Konfiguriere deinen TTS-API-Token.",
        "diagnostics": "Diagnose",
        "stop": "Nimm eine Antwort auf. Dein Coach antwortet automatisch.",
        "no_speech": "Keine Sprache erkannt. Bitte versuche es erneut.",
        "reset": "Neue Übungssitzung starten",
        "history": "Gesprächsverlauf",
        "no_models": "Installierte Ollama-Modelle konnten nicht abgerufen werden.",
        "tts_error": "TTS-Generierung fehlgeschlagen",
        "ollama_error": "Ollama-Anfrage fehlgeschlagen",
        "asr_error": "Parakeet-Transkription fehlgeschlagen",
        "session_saved": "Gespräch gespeichert.",
        "download": "Gespräch herunterladen",
        "listening": "Bereit für deine nächste Antwort.",
    },
}

# ============================================================
# CONSTANTS
# ============================================================

PRACTICE_LANGUAGES = ["English", "German"]

LEVELS = [
    "A2",
    "B1",
    "B1-B2",
    "B2",
    "C1",
]

FOCUS_OPTIONS = [
    "Fluency and confidence",
    "Grammar and sentence structure",
    "Vocabulary expansion",
    "Pronunciation awareness",
    "Professional communication",
    "Job interviews",
    "Listening comprehension",
]

SCENARIOS = [
    "Free conversation",
    "Daily life",
    "Workplace small talk",
    "Technical software engineering discussion",
    "Customer meeting",
    "Job interview",
    "Presentation",
    "Travel and appointments",
    "Business negotiation",
]

CORRECTION_LEVELS = [
    "Light",
    "Balanced",
    "Detailed",
]

RESPONSE_LENGTHS = [
    "Short",
    "Medium",
    "Detailed",
]

# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    defaults = {
        "messages": [],
        "turns": 0,
        "words": 0,
        "corrections": 0,
        "audio_data": None,
        "last_spoken": "",
        "last_audio_signature": "",
        "session_id": str(uuid.uuid4())[:8],
        "session_started": datetime.now().isoformat(),
        "pending_user_text": None,
        "pending_source": None,
        "pending_transcript": None,
        "last_error": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()

# ============================================================
# SESSION MANAGEMENT
# ============================================================

def reset_session():

    st.session_state.messages = []
    st.session_state.turns = 0
    st.session_state.words = 0
    st.session_state.corrections = 0

    st.session_state.audio_data = None
    st.session_state.last_spoken = ""
    st.session_state.last_audio_signature = ""

    st.session_state.pending_user_text = None
    st.session_state.pending_source = None
    st.session_state.pending_transcript = None

    st.session_state.last_error = None

    st.session_state.session_id = str(uuid.uuid4())[:8]

    st.session_state.session_started = (
        datetime.now().isoformat()
    )


def save_session(settings):

    record = {
        "session_id": st.session_state.session_id,
        "started_at": st.session_state.session_started,
        "updated_at": datetime.now().isoformat(),
        "settings": settings,
        "statistics": {
            "turns": st.session_state.turns,
            "words": st.session_state.words,
            "corrections": st.session_state.corrections,
        },
        "messages": st.session_state.messages,
    }

    path = SESSION_DIR / (
        f"session_{st.session_state.session_id}.json"
    )

    path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")

    interface_language = st.selectbox(
        "Interface Language",
        ["English", "Deutsch"],
        key="interface_language",
    )

    L = LABELS[interface_language]

    st.header(L["settings"])

    practice_language = st.selectbox(
        L["language"],
        PRACTICE_LANGUAGES,
        key="practice_language",
    )

    level = st.selectbox(
        L["level"],
        LEVELS,
        index=2,
        key="learner_level",
    )

    focus = st.selectbox(
        L["focus"],
        FOCUS_OPTIONS,
        key="learning_focus",
    )

    scenario = st.selectbox(
        L["scenario"],
        SCENARIOS,
        key="scenario",
    )

    correction_intensity = st.select_slider(
        L["correction"],
        options=CORRECTION_LEVELS,
        value="Balanced",
        key="correction_intensity",
    )

    feedback_language = st.selectbox(
        L["feedback"],
        [
            "Same as practice language",
            "English",
            "German",
        ],
        key="feedback_language",
    )

    response_length = st.select_slider(
        L["length"],
        options=RESPONSE_LENGTHS,
        value="Medium",
        key="response_length",
    )

    personal_goal = st.text_area(
        L["goal"],
        value=(
            "I want to speak more naturally and confidently. "
            "Help me improve grammar, sentence structure, "
            "and active vocabulary."
        ),
        height=110,
        key="personal_goal",
    )

    # --------------------------------------------------------
    # OLLAMA SETTINGS
    # --------------------------------------------------------

    st.divider()
    st.subheader("🧠 Ollama")

    ollama_url = st.text_input(
        L["url"],
        value=os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
        ),
        key="ollama_url",
    )

    installed_models = get_ollama_models(
        ollama_url
    )

    if installed_models:

        st.caption(
            "Installed: " + ", ".join(installed_models)
        )

    else:

        st.caption(L["no_models"])

    ollama_model = st.text_input(
        L["model"],
        value=os.getenv(
            "OLLAMA_MODEL",
            "llama3.2",
        ),
        key="ollama_model",
        help=(
            "Enter any installed Ollama model, "
            "for example llama3.2 or qwen3:4b."
        ),
    )

    if st.button(
        L["refresh"],
        use_container_width=True,
    ):

        st.rerun()

    temperature = st.slider(
        L["temperature"],
        min_value=0.1,
        max_value=1.2,
        value=0.65,
        step=0.05,
        key="temperature",
    )

    # --------------------------------------------------------
    # SPEECH SETTINGS
    # --------------------------------------------------------

    st.divider()
    st.subheader("🎤 Speech Settings")

    asr_model = st.text_input(
        L["asr"],
        value=os.getenv(
            "PARAKEET_MODEL",
            "nvidia/parakeet-tdt-0.6b-v3",
        ),
        key="asr_model",
    )

    max_cpu = max(
        2,
        os.cpu_count() or 4,
    )

    cpu_threads = st.slider(
        L["threads"],
        1,
        max_cpu,
        min(4, max_cpu),
        key="cpu_threads",
    )

    tts_url = st.text_input(
        L["tts_url"],
        value=os.getenv(
            "TTS_BASE_URL",
            "http://127.0.0.1:7849/v1",
        ),
        key="tts_url",
    )

    tts_model = st.text_input(
        L["tts_model"],
        value=os.getenv(
            "TTS_MODEL",
            "phonon-40m-multilingual",
        ),
        key="tts_model",
    )

    voice = st.text_input(
        L["voice"],
        value=os.getenv(
            "TTS_VOICE",
            "Anton",
        ),
        key="tts_voice",
        help=(
            "Examples: Annike or Anton. "
            "Use the exact voice ID accepted by TTS Studio."
        ),
    )

    tts_token = st.text_input(
        L["token"],
        value=os.getenv("TTS_API_KEY", ""),
        type="password",
        key="tts_token",
    )

    # --------------------------------------------------------
    # PRIVACY AND SESSION SETTINGS
    # --------------------------------------------------------

    st.divider()

    save_locally = st.checkbox(
        L["save"],
        value=True,
        key="save_locally",
    )

    show_transcript = st.checkbox(
        L["show_transcript"],
        value=True,
        key="show_transcript",
    )

    if st.button(
        L["reset"],
        use_container_width=True,
    ):

        reset_session()
        st.rerun()


L = LABELS[interface_language]

# ============================================================
# CURRENT SETTINGS SNAPSHOT
# ============================================================

current_settings = {
    "practice_language": practice_language,
    "interface_language": interface_language,
    "level": level,
    "focus": focus,
    "scenario": scenario,
    "correction_intensity": correction_intensity,
    "feedback_language": feedback_language,
    "response_length": response_length,
    "ollama_model": ollama_model,
    "ollama_url": ollama_url,
    "asr_model": asr_model,
    "tts_model": tts_model,
    "tts_voice": voice,
}

# ============================================================
# HEADER
# ============================================================

st.title("🎙️ " + L["title"])

st.caption(L["subtitle"])

st.info("🔄 " + L["stop"])

# ============================================================
# STATISTICS
# ============================================================

st.subheader("📊 " + (
    "Session Progress"
    if interface_language == "English"
    else "Lernfortschritt"
))

col1, col2, col3 = st.columns(3)

col1.metric(
    L["turns"],
    st.session_state.turns,
)

col2.metric(
    L["words"],
    st.session_state.words,
)

col3.metric(
    L["corrections"],
    st.session_state.corrections,
)

st.divider()

# ============================================================
# CONVERSATION DISPLAY
# ============================================================

st.subheader("💬 " + L["conversation"])

if not st.session_state.messages:

    st.info(L["empty"])

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.markdown(message["content"])

            if (
                message.get("source") == "voice"
                and show_transcript
            ):

                st.caption(
                    "🎤 " + str(
                        message.get("transcript", "")
                    )
                )

        else:

            st.markdown(
                message.get("reply", "")
            )

            if message.get("correction"):

                with st.expander(
                    "✍️ " + L["correction"]
                ):

                    st.markdown(
                        message["correction"]
                    )

                    if message.get("explanation"):

                        st.markdown(
                            "**" + L["explanation"] + "**"
                        )

                        st.write(
                            message["explanation"]
                        )

            if message.get("vocabulary"):

                with st.expander(
                    "📚 " + L["vocabulary"]
                ):

                    st.markdown(
                        message["vocabulary"]
                    )

# ============================================================
# CONVERSATION ENGINE
# ============================================================

def process_turn(
    user_text,
    source="text",
    transcript=None,
):

    if not user_text or not user_text.strip():

        st.warning(L["no_speech"])
        return

    user_text = user_text.strip()

    # Add user message to conversation.

    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "source": source,
        "transcript": transcript,
    })

    st.session_state.words += len(
        re.findall(
            r"\b[\w'-]+\b",
            user_text,
        )
    )

    st.session_state.turns += 1

    # Feedback language.

    if feedback_language == "Same as practice language":

        selected_feedback_language = practice_language

    else:

        selected_feedback_language = feedback_language

    # Build system prompt.

    system_prompt = build_system_prompt(
        language=practice_language,
        level=level,
        focus=focus,
        scenario=scenario,
        correction_intensity=correction_intensity,
        feedback_language=selected_feedback_language,
        personal_goal=personal_goal,
        response_length=response_length,
    )

    # Prepare previous conversation.

    history = []

    for item in st.session_state.messages[:-1]:

        if item["role"] == "user":

            history.append({
                "role": "user",
                "content": item["content"],
            })

        else:

            history.append({
                "role": "assistant",
                "content": item["reply"],
            })

    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    try:

        with st.spinner(L["thinking"]):

            coach = CoachClient(
                base_url=ollama_url,
                model=ollama_model,
                temperature=temperature,
            )

            raw_response = coach.reply(
                system_prompt=system_prompt,
                history=history,
                user_text=user_text,
            )

            result = parse_coach_response(
                raw_response
            )

    except Exception as exc:

        st.session_state.messages.pop()
        st.session_state.turns -= 1

        st.error(
            f"{L['ollama_error']}: {exc}"
        )

        return

    # --------------------------------------------------------
    # SAVE COACH RESPONSE
    # --------------------------------------------------------

    assistant_message = {
        "role": "assistant",
        "reply": result["reply"],
        "correction": result["correction"],
        "explanation": result["explanation"],
        "vocabulary": result["vocabulary"],
    }

    st.session_state.messages.append(
        assistant_message
    )

    if result["correction"]:

        st.session_state.corrections += 1

    # --------------------------------------------------------
    # AUTOMATIC TTS GENERATION
    # --------------------------------------------------------

    st.session_state.audio_data = None
    st.session_state.last_spoken = ""

    if tts_token:

        try:

            with st.spinner(L["speaking"]):

                audio = synthesize_speech(
                    text=result["reply"],
                    base_url=tts_url,
                    api_key=tts_token,
                    model=tts_model,
                    voice=voice,
                )

                st.session_state.audio_data = audio
                st.session_state.last_spoken = result["reply"]

        except Exception as exc:

            st.warning(
                f"{L['tts_error']}: {exc}"
            )

    else:

        st.warning(
            L["missing_token"]
        )

    # --------------------------------------------------------
    # SAVE SESSION
    # --------------------------------------------------------

    if save_locally:

        save_session(current_settings)

    # Re-render after processing.
    st.rerun()


# ============================================================
# MICROPHONE INPUT
# ============================================================

st.divider()

st.subheader("🎤 " + L["record"])

st.caption(L["record_help"])

audio_input = st.audio_input(
    L["record"],
    sample_rate=16000,
    key=f"microphone_{st.session_state.session_id}",
)

send_voice = st.button(
    L["send_voice"],
    type="primary",
    disabled=audio_input is None,
    use_container_width=True,
    key="send_voice",
)

if send_voice and audio_input is not None:

    try:

        with st.spinner(L["transcribing"]):

            transcript = transcribe_parakeet(
                audio_bytes=audio_input.getvalue(),
                model_name=asr_model,
                cpu_threads=cpu_threads,
            )

        if transcript.strip():

            st.success(
                f"**{L['you']}:** {transcript}"
            )

            process_turn(
                user_text=transcript,
                source="voice",
                transcript=transcript,
            )

        else:

            st.warning(L["no_speech"])

    except Exception as exc:

        st.error(
            f"{L['asr_error']}: {exc}"
        )

# ============================================================
# TEXT INPUT
# ============================================================

st.divider()

st.subheader("⌨️ " + L["type"])

with st.form(
    "text_input_form",
    clear_on_submit=True,
):

    typed_text = st.text_area(
        "Your answer",
        height=100,
        placeholder="Write your answer here...",
        label_visibility="collapsed",
    )

    send_text = st.form_submit_button(
        L["send_text"],
        use_container_width=True,
    )

    if send_text and typed_text.strip():

        process_turn(
            user_text=typed_text.strip(),
            source="text",
        )

# ============================================================
# AUTOMATIC AUDIO PLAYER
# ============================================================

if st.session_state.audio_data:

    st.divider()

    st.subheader(
        "🔊 " + L["coach"]
    )

    st.audio(
        st.session_state.audio_data,
        format="audio/mpeg",
        autoplay=True,
    )

    st.caption(L["autoplay"])

# ============================================================
# CONVERSATION ACTIONS
# ============================================================

if st.session_state.messages:

    last_message = st.session_state.messages[-1]

    if last_message["role"] == "assistant":

        st.divider()

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # REPEAT RESPONSE
        # ----------------------------------------------------

        with col1:

            if st.button(
                "🔁 " + L["repeat"],
                use_container_width=True,
            ):

                try:

                    with st.spinner(L["speaking"]):

                        audio = synthesize_speech(
                            text=last_message["reply"],
                            base_url=tts_url,
                            api_key=tts_token,
                            model=tts_model,
                            voice=voice,
                        )

                    st.session_state.audio_data = audio

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"{L['tts_error']}: {exc}"
                    )

        # ----------------------------------------------------
        # SIMPLIFY RESPONSE
        # ----------------------------------------------------

        with col2:

            if st.button(
                "💡 " + L["simplify"],
                use_container_width=True,
            ):

                process_turn(
                    user_text=(
                        "Please explain your previous response "
                        "more simply, using vocabulary appropriate "
                        "for my language level. Continue naturally "
                        "with one follow-up question."
                    ),
                    source="text",
                )

# ============================================================
# SESSION DOWNLOAD
# ============================================================

if st.session_state.messages:

    with st.expander(
        "📥 " + L["history"]
    ):

        session_data = {
            "session_id": st.session_state.session_id,
            "started_at": st.session_state.session_started,
            "settings": current_settings,
            "statistics": {
                "turns": st.session_state.turns,
                "words": st.session_state.words,
                "corrections": st.session_state.corrections,
            },
            "messages": st.session_state.messages,
        }

        st.download_button(
            L["download"],
            data=json.dumps(
                session_data,
                ensure_ascii=False,
                indent=2,
            ),
            file_name=(
                f"speaking_session_"
                f"{st.session_state.session_id}.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

# ============================================================
# DIAGNOSTICS
# ============================================================

with st.expander(
    "🧰 " + L["diagnostics"]
):

    st.write("Ollama endpoint:", ollama_url)
    st.write("Ollama model:", ollama_model)

    st.write("Parakeet model:", asr_model)

    st.write("TTS endpoint:", tts_url)
    st.write("TTS model:", tts_model)
    st.write("TTS voice:", voice)

    st.write(
        "TTS token configured:",
        bool(tts_token),
    )

    st.write(
        "Session ID:",
        st.session_state.session_id,
    )

    st.write(
        "Session storage:",
        str(SESSION_DIR),
    )

    st.write(
        "Messages:",
        len(st.session_state.messages),
    )

    st.write(
        "CPU threads:",
        cpu_threads,
    )

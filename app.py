import os
import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from rag import retrieve_context
from memory import ( init_db, extract_long_term_memories, save_memory, search_memory, list_memories, delete_memory, get_latest_memory_value,)
from prompts import build_prompt, format_chat_history, format_memories_for_prompt
from timeline import load_timeline, format_timeline_for_prompt
from voice_clone import ( clone_voice_to_wav_bytes, has_reference_voice, load_chatterbox_model,)

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

APP_TITLE = "Digital Twin of Andrew Ng"

@st.cache_resource(show_spinner="Loading the local voice model...")
def get_voice_clone_model():
    return load_chatterbox_model()

def transcribe_audio_bytes(audio_bytes, suffix):
    if sr is None:
        return "", "SpeechRecognition is not installed."

    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text.strip(), ""
    except sr.UnknownValueError:
        return "", "Could not understand the audio."
    except sr.RequestError as exc:
        return "", f"Speech recognition error: {exc}"
    except Exception as exc:
        return "", str(exc)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

async def _tts_to_file(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(output_path)

def tts_audio_bytes(text, voice="en-US-GuyNeural"):
    if edge_tts is None:
        return b"", "edge-tts is not installed."
    if not text.strip():
        return b"", "No text to speak."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        output_path = tmp.name

    try:
        asyncio.run(_tts_to_file(text, voice, output_path))
        with open(output_path, "rb") as f:
            return f.read(), ""
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass

def call_gemini(prompt, api_key, model_name, temperature):
    if not api_key:
        return "Setup error: GEMINI_API_KEY is not set."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": temperature},
    )
    response = model.generate_content(prompt)
    return response.text or ""

def determine_skill_level(override):
    if override != "Auto":
        return override
    stored = get_latest_memory_value("skill_level")
    return stored or "Intermediate"

def generate_response(user_query, chat_history, settings):
    context_text, sources, chunks = retrieve_context(
        user_query, k=settings["top_k"]
    )
    if not context_text:
        context_text = "No sources available."

    memory_hits = search_memory(user_query, limit=5)
    memories_text = format_memories_for_prompt(memory_hits)
    timeline_text = format_timeline_for_prompt(load_timeline())
    history_text = format_chat_history(chat_history, max_turns=settings["history_turns"])

    skill_level = determine_skill_level(settings["skill_level"])
    prompt = build_prompt(
        user_query=user_query,
        chat_history=history_text,
        retrieved_context=context_text,
        memories=memories_text,
        timeline=timeline_text,
        skill_level=skill_level,
    )

    response_text = call_gemini(
        prompt,
        settings["api_key"],
        settings["model_name"],
        settings["temperature"],
    ).strip()

    if not response_text:
        response_text = "I do not have enough sources to answer that yet."

    for key, value, importance in extract_long_term_memories(user_query, settings["api_key"]):
        save_memory(key, value, importance)

    return response_text, sources, chunks, memory_hits

def render_sources(sources):
    if not sources:
        st.write("No sources retrieved yet.")
        return
    for source in sources:
        title = source.get("title", "").strip()
        label = f"[{source['index']}] {title}" if title else f"[{source['index']}]"
        st.write(label)
        if source.get("source"):
            st.write(f"Source: {source['source']}")
        if source.get("url"):
            st.write(f"URL: {source['url']}")
        st.write("---")

def render_chunks(chunks):
    if not chunks:
        st.write("No chunks retrieved yet.")
        return
    for idx, chunk in enumerate(chunks, start=1):
        st.write(f"Chunk {idx}")
        st.write(chunk.get("text", "")[:800])
        if chunk.get("source"):
            st.write(f"Source: {chunk['source']}")
        if chunk.get("title"):
            st.write(f"Title: {chunk['title']}")
        st.write("---")

def render_memory_table(memories):
    if not memories:
        st.write("No memories stored yet.")
        return
    st.dataframe(memories, use_container_width=True)

def render_chat_page(settings):
    st.title(APP_TITLE)
    st.caption("Learn machine learning with Andrew Ng's teaching style.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.expander("Voice Interaction", expanded=True):
        st.caption("Record a question, then send it through the chat.")
        recording = st.audio_input("Record your question", sample_rate=16000)
        upload = st.file_uploader(
            "Or upload audio (wav or flac)",
            type=["wav", "flac"],
        )
        voice_input = recording or upload
        if voice_input is not None and sr is not None:
            if st.button("Transcribe and Send", type="primary"):
                suffix = os.path.splitext(voice_input.name)[1] or ".wav"
                transcript, error = transcribe_audio_bytes(voice_input.getvalue(), suffix)
                if error:
                    st.error(error)
                elif transcript.strip():
                    if not settings["api_key"]:
                        st.error("Set GEMINI_API_KEY to use Gemini.")
                    else:
                        st.session_state.messages.append({"role": "user", "content": transcript})
                        with st.spinner("Thinking..."):
                            response_text, sources, chunks, _ = generate_response(
                                transcript,
                                st.session_state.messages,
                                settings,
                            )
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.session_state.last_sources = sources
                        st.session_state.last_chunks = chunks
                        st.session_state.last_response = response_text
                        st.session_state.speak_last_response = settings["auto_speak"]
                        st.rerun()

    user_query = st.chat_input("Your message...")
    if user_query and user_query.strip():
        if not settings["api_key"]:
            st.error("Set GEMINI_API_KEY to use Gemini.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.spinner("Thinking..."):
                response_text, sources, chunks, _ = generate_response(
                    user_query,
                    st.session_state.messages,
                    settings,
                )
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.last_sources = sources
            st.session_state.last_chunks = chunks
            st.session_state.last_response = response_text
            st.session_state.speak_last_response = settings["auto_speak"]
            st.rerun()

    with st.expander("Sources panel", expanded=False):
        render_sources(st.session_state.last_sources)

    with st.expander("Retrieved chunks panel", expanded=False):
        render_chunks(st.session_state.last_chunks)

    if st.session_state.last_response:
        st.write("Generate a voice reply:")
        voice_mode = settings["voice_mode"]
        if st.button("Speak reply"):
            st.session_state.speak_last_response = True

        if st.session_state.speak_last_response:
            try:
                if voice_mode == "My cloned voice":
                    model = get_voice_clone_model()
                    audio_bytes = clone_voice_to_wav_bytes(
                        model, st.session_state.last_response
                    )
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                else:
                    audio_bytes, error = tts_audio_bytes(
                        st.session_state.last_response,
                        voice=settings["edge_voice"],
                    )
                    if error:
                        raise RuntimeError(error)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception as exc:
                st.error(f"Could not generate voice reply: {exc}")
            finally:
                st.session_state.speak_last_response = False

def render_memory_dashboard():
    st.title("Memory Dashboard")

    query = st.text_input("Search memories")
    if query:
        memories = search_memory(query, limit=200)
    else:
        memories = list_memories(limit=200)
    render_memory_table(memories)

    st.subheader("Add memory")
    with st.form("add_memory"):
        key = st.text_input("Key")
        value = st.text_area("Value")
        importance = st.slider("Importance", 0.0, 1.0, 0.5, 0.05)
        submitted = st.form_submit_button("Save")
        if submitted:
            save_memory(key, value, importance)
            st.success("Memory saved.")

    st.subheader("Delete memory")
    delete_id = st.text_input("Memory ID to delete")
    if st.button("Delete"):
        try:
            deleted = delete_memory(int(delete_id))
        except ValueError:
            deleted = False
        if deleted:
            st.success("Memory deleted.")
        else:
            st.error("Memory not found or invalid ID.")

def main():
    load_dotenv(override=True)
    init_db()

    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_chunks" not in st.session_state:
        st.session_state.last_chunks = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "speak_last_response" not in st.session_state:
        st.session_state.speak_last_response = False

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    page = st.sidebar.radio("Page", ["Chat", "Memory Dashboard"])

    with st.sidebar.expander("Settings", expanded=True):
        model_name = st.text_input("Gemini model", value="gemini-3.1-flash-lite")
        top_k = st.slider("Top-k retrieved chunks", 1, 10, 5)
        history_turns = st.slider("History turns", 2, 12, 6)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
        profile_choice = st.selectbox(
            "Learning profile",
            ["Auto", "Beginner", "Intermediate", "Advanced"],
        )
        if profile_choice != "Auto" and st.button("Save profile"):
            save_memory("skill_level", profile_choice, 0.9)
            st.success("Saved skill level to long-term memory.")

    with st.sidebar.expander("Voice Clone", expanded=True):
        voice_mode = st.radio(
            "Reply voice",
            ["My cloned voice", "Standard voice"],
            index=0 if has_reference_voice() else 1,
        )
        auto_speak = st.toggle("Speak replies automatically", value=True)

        edge_voice = st.selectbox(
            "Standard voice",
            ["en-US-GuyNeural", "en-US-AriaNeural", "en-US-JennyNeural"],
        )

        if has_reference_voice():
            st.success("Voice reference file found in database/voices/my_voice.wav")
        else:
            st.info("Place your WAV voice reference file at database/voices/my_voice.wav to enable voice cloning.")

    with st.sidebar.expander("Memory Viewer", expanded=False):
        render_memory_table(list_memories(limit=10))

    with st.sidebar.expander("Timeline Viewer", expanded=False):
        timeline = load_timeline()
        if timeline:
            st.table(timeline)
        else:
            st.write("Timeline data not found.")

    settings = {
        "api_key": api_key,
        "model_name": model_name,
        "top_k": top_k,
        "history_turns": history_turns,
        "temperature": temperature,
        "skill_level": profile_choice,
        "chroma_dir": "database/chroma",
        "voice_mode": voice_mode,
        "auto_speak": auto_speak,
        "edge_voice": edge_voice,
    }

    if page == "Memory Dashboard":
        render_memory_dashboard()
    else:
        render_chat_page(settings)

if __name__ == "__main__":
    main()

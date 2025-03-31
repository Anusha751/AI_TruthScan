import streamlit as st
import base64
import requests
import os
import numpy as np
import sounddevice as sd
import wavio
import threading
import speech_recognition as sr

# -------------------- CONFIGURATION --------------------
# Flask API URL
FLASK_API_URL = "http://localhost:5000"

# Supported language (English only)
SUPPORTED_LANGUAGES = {"en": "English"}

# -------------------- SESSION STATE --------------------
if "recording_audio" not in st.session_state:
    st.session_state["recording_audio"] = False
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""
if "gaze_percentage" not in st.session_state:
    st.session_state["gaze_percentage"] = 0.0
if "audio_file" not in st.session_state:
    st.session_state["audio_file"] = None

# -------------------- TITLE --------------------
st.title("🎙️ AI-TruthScan: Detect AI-Generated Interview Responses")

# -------------------- UPLOAD SECTION --------------------
with st.expander("📁 Upload Audio or Video File", expanded=True):
    upload_option = st.radio("Select file type to upload", ("Audio", "Video"))
    if upload_option == "Audio":
        uploaded_file = st.file_uploader("Upload an audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.session_state["audio_file"] = audio_bytes
            b64_audio = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio controls>
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
            """
            st.components.v1.html(audio_html, height=70)
    else:
        uploaded_file = st.file_uploader("Upload a video file (mp4, mov)", type=["mp4", "mov"])
        if uploaded_file:
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state["video_file"] = video_path
            st.session_state["audio_file"] = None

# -------------------- RECORDING SECTION --------------------
with st.expander("🎧 Live Audio Recording", expanded=True):
    if st.button("Start Audio Recording"):
        st.session_state["recording_audio"] = True
        st.write("Recording audio...")

        def record_audio():
            fs = 44100  # Sample rate
            recording = []
            with sd.InputStream(samplerate=fs, channels=1) as stream:
                while st.session_state["recording_audio"]:
                    data = stream.read(fs // 10)[0]  # Read in chunks
                    recording.append(data)
            audio_data = np.concatenate(recording, axis=0)
            wav_path = "temp_audio.wav"
            # Save audio as WAV file
            wavio.write(wav_path, audio_data, fs, sampwidth=2)
            with open(wav_path, "rb") as f:
                st.session_state["audio_file"] = f.read()
            os.remove(wav_path)
            st.session_state["recording_audio"] = False
            st.write("Audio recording complete.")

        threading.Thread(target=record_audio).start()

    if st.button("Stop Audio Recording"):
        st.session_state["recording_audio"] = False

# -------------------- ANALYSIS SECTION --------------------
with st.expander("🧠 Analysis Settings", expanded=True):
    context = st.selectbox("Select the context of the speech", ["General", "Casual Conversation", "Formal Interview"])
    threshold = st.slider("Set AI-Generated Probability Threshold (%)", 0, 100, 50)

if st.button("🧠 Analyze"):
    if st.session_state["audio_file"]:
        with st.spinner("Analyzing..."):
            files = {"file": ("audio.wav", st.session_state["audio_file"], "audio/wav")}
            data = {
                "file_type": "audio",
                "context": context,
                "threshold": str(threshold)
            }

            try:
                response = requests.post(f"{FLASK_API_URL}/analyze", files=files, data=data)
                result = response.json()
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.session_state["transcription"] = result["transcription"]
                    st.session_state["gaze_percentage"] = result["gaze_percentage"]

                    st.subheader("📊 Analysis Result")
                    st.markdown(f"**Transcription:**\n\n{result['transcription']}")
                    st.markdown(f"**Gaze Analysis:** Looking away {result['gaze_percentage']:.2f}% of the time")
                    st.markdown("---")
                    st.markdown(f"**Classification:** {result['classification']}")
                    st.markdown(f"**Probabilities:** Real (Human-Created) - {result['human_prob']}% | Fake (AI-Generated) - {result['ai_prob']}%")
                    st.markdown(f"**Justification:**\n\n{result['justification']}")
            except Exception as e:
                st.error(f"Error analyzing file: {e}")
    else:
        st.warning("⚠️ No audio file available for analysis.")

# -------------------- VIEW SAVED RESULTS --------------------
with st.expander("📂 View Saved Results"):
    try:
        response = requests.get(f"{FLASK_API_URL}/results")
        results = response.json()
        for result in results:
            st.markdown(f"**ID:** {result['id']} | **Timestamp:** {result['timestamp']}")
            st.markdown(f"**Transcription:** {result['transcription']}")
            st.markdown(f"**Classification:** {result['classification']}")
            st.markdown(f"**Probabilities:** Real - {result['human_prob']}% | Fake - {result['ai_prob']}%")
            st.markdown(f"**Justification:** {result['justification']}")
            st.markdown("---")
    except Exception as e:
        st.error(f"Error fetching saved results: {e}")

# -------------------- DISCLAIMER --------------------
st.markdown("---")
st.info("⚠️ *Note:* Results are estimates based on patterns. Use as a guide.")

import streamlit as st
import requests
import base64
import threading
import time
import sounddevice as sd
import wavio
import cv2
import streamlit.components.v1 as components

# Flask API URL
API_URL = "http://localhost:5000"

# -------------------- SESSION STATE --------------------
if "recording_audio" not in st.session_state:
    st.session_state["recording_audio"] = False
if "recording_video" not in st.session_state:
    st.session_state["recording_video"] = False
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# -------------------- TITLE --------------------
st.title("🎙️ AI-TruthScan: Detect AI-Generated Interview Responses")

# -------------------- UPLOAD SECTION --------------------
with st.expander("📁 Upload Audio or Video File", expanded=True):
    upload_option = st.radio("Select file type to upload", ("Audio", "Video"))
    uploaded_file = st.file_uploader(f"Upload an {upload_option.lower()} file", 
                                    type=["mp3", "wav", "m4a"] if upload_option == "Audio" else ["mp4", "mov"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        b64_audio = base64.b64encode(file_bytes).decode()
        if upload_option == "Audio":
            audio_html = f"""
            <audio controls>
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
            """
            components.html(audio_html, height=70)

# -------------------- RECORDING SECTION --------------------
with st.expander("🎧 Live Recording", expanded=True):
    record_option = st.radio("Select recording type", ("Audio", "Video"))
    
    if record_option == "Audio":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Audio Recording"):
                st.session_state["recording_audio"] = True
                def record_audio():
                    fs = 44100
                    duration = 10
                    st.session_state["status"] = "Recording audio..."
                    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                    sd.wait()
                    wavio.write("temp_audio.wav", recording, fs, sampwidth=2)
                    with open("temp_audio.wav", "rb") as f:
                        audio_bytes = f.read()
                    st.session_state["audio_bytes"] = audio_bytes.hex()
                    os.remove("temp_audio.wav")
                    st.session_state["recording_audio"] = False
                    st.session_state["status"] = "Recording complete."
                threading.Thread(target=record_audio).start()
        with col2:
            if st.button("Stop Audio Recording"):
                st.session_state["recording_audio"] = False
        if "status" in st.session_state:
            st.write(st.session_state["status"])
    
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Video Recording"):
                st.session_state["recording_video"] = True
                def record_video():
                    cap = cv2.VideoCapture(0)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter("temp_video.mp4", fourcc, 20.0, (640, 480))
                    st.session_state["status"] = "Recording video..."
                    start_time = time.time()
                    while time.time() - start_time < 10:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                    cap.release()
                    out.release()
                    with open("temp_video.mp4", "rb") as f:
                        video_bytes = f.read()
                    st.session_state["video_bytes"] = video_bytes.hex()
                    os.remove("temp_video.mp4")
                    st.session_state["recording_video"] = False
                    st.session_state["status"] = "Recording complete."
                threading.Thread(target=record_video).start()
        with col2:
            if st.button("Stop Video Recording"):
                st.session_state["recording_video"] = False
        if "status" in st.session_state:
            st.write(st.session_state["status"])

# -------------------- ANALYSIS SECTION --------------------
with st.expander("🧠 Analysis Settings", expanded=True):
    context = st.selectbox("Select the context of the speech", ["General", "Casual Conversation", "Formal Interview"])
    threshold = st.slider("Set AI-Generated Probability Threshold (%)", 0, 100, 50)

if st.button("🧠 Analyze"):
    with st.spinner("Analyzing..."):
        data = {"context": context, "threshold": threshold}
        if "audio_bytes" in st.session_state:
            data["audio_bytes"] = st.session_state["audio_bytes"]
        elif "video_bytes" in st.session_state:
            data["video_bytes"] = st.session_state["video_bytes"]
        elif uploaded_file:
            data["audio_bytes" if upload_option == "Audio" else "video_bytes"] = file_bytes.hex()
        else:
            st.warning("⚠️ No file or recording available.")
            st.stop()

        response = requests.post(f"{API_URL}/analyze", json=data)
        if response.status_code == 200:
            st.session_state["analysis_result"] = response.json()
            result = st.session_state["analysis_result"]
            st.subheader("📊 Analysis Result")
            st.markdown(f"**Transcription:**\n\n{result['transcription']}")
            st.markdown(f"**Gaze Analysis:** Looking away {result['gaze_percentage']:.2f}% of the time")
            st.markdown("---")
            st.markdown(f"**Classification:** {result['classification']}")
            st.markdown(f"**Probabilities:** Real (Human-Created) - {result['human_prob']}% | Fake (AI-Generated) - {result['ai_prob']}%")
            st.markdown(f"**Justification:**\n\n{result['justification']}")
            st.success("✅ Results saved to database!")
        else:
            st.error(f"⚠️ Analysis failed: {response.json().get('error')}")

# -------------------- VIEW SAVED RESULTS --------------------
with st.expander("📂 View Saved Results"):
    response = requests.get(f"{API_URL}/results")
    if response.status_code == 200:
        results = response.json()
        for row in results:
            st.markdown(f"**ID:** {row['id']} | **Timestamp:** {row['timestamp']}")
            st.markdown(f"**Transcription:** {row['transcription']}")
            st.markdown(f"**Classification:** {row['classification']}")
            st.markdown(f"**Probabilities:** Real - {row['human_prob']}% | Fake - {row['ai_prob']}%")
            st.markdown(f"**Justification:** {row['justification']}")
            st.markdown("---")
    else:
        st.error("⚠️ Could not retrieve saved results.")

# -------------------- DISCLAIMER --------------------
st.markdown("---")
st.info("⚠️ *Note:* Results are estimates based on patterns. Use as a guide.")

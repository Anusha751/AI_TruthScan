from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import speech_recognition as sr
import cv2
import numpy as np
import mediapipe as mp
import re
import sqlite3
import os
import ffmpeg
from io import BytesIO
import time

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from Streamlit (or other frontends)

# -------------------- CONFIGURATION --------------------
GOOGLE_API_KEY = "api_key"  # Replace with your actual Google API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# MediaPipe setup for gaze tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Database setup (SQLite)
def init_db():
    conn = sqlite3.connect("analysis_results.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  transcription TEXT,
                  classification TEXT,
                  human_prob REAL,
                  ai_prob REAL,
                  justification TEXT,
                  timestamp TEXT)''')
    conn.commit()
    return conn

# -------------------- PREPROCESSING FUNCTIONS --------------------
def count_filler_words(transcription):
    """Count filler words typical in human speech."""
    filler_words = ["um", "uh", "like", "you know", "er", "well"]
    pattern = r'\b(' + '|'.join(filler_words) + r')\b'
    return len(re.findall(pattern, transcription.lower(), re.IGNORECASE))

def extract_audio_from_video(video_path):
    """Extract audio from a video file and return as bytes."""
    output_audio_path = "temp_audio.wav"
    try:
        ffmpeg.input(video_path).output(output_audio_path, format='wav').run(overwrite_output=True, quiet=True)
        with open(output_audio_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(output_audio_path)
        return audio_bytes
    except Exception as e:
        return None

def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text."""
    recognizer = sr.Recognizer()
    audio_data = BytesIO(audio_bytes)
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="en")
        except sr.UnknownValueError:
            return "❌ Could not understand the audio."
        except sr.RequestError:
            return "❌ Error with the speech recognition service."

def analyze_gaze(video_path):
    """Analyze gaze direction using MediaPipe to detect if the person is looking away."""
    cap = cv2.VideoCapture(video_path)
    looking_away_count = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_iris = face_landmarks.landmark[468]  # Left iris center
                right_iris = face_landmarks.landmark[473]  # Right iris center
                nose_tip = face_landmarks.landmark[1]  # Nose tip as face center
                left_gaze_offset = abs(left_iris.x - nose_tip.x)
                right_gaze_offset = abs(right_iris.x - nose_tip.x)
                if left_gaze_offset > 0.1 or right_gaze_offset > 0.1:
                    looking_away_count += 1

    cap.release()
    looking_away_percentage = (looking_away_count / total_frames) * 100 if total_frames > 0 else 0
    return looking_away_percentage

def get_gemini_response(transcription, gaze_percentage, context="Formal Interview"):
    """Get analysis from Gemini API."""
    filler_count = count_filler_words(transcription)
    prompt = f"""
    You are an advanced AI content analyzer designed to detect AI-generated responses in spoken text during interviews. 
    Your goal is to differentiate between human-spoken responses and AI-generated ones.

    Consider the following factors while analyzing:
    - Linguistic Analysis:
      - Human traits: filler words (e.g., "um," "uh"), self-corrections, personal anecdotes, emotional tone, slight grammatical errors, or context-specific phrasing (e.g., nervousness in interviews).
      - AI traits: overly formal tone, perfect grammar, repetitive structure, lack of personal depth, generic phrasing.
    - Behavioral Analysis:
      - Gaze tracking: If the person is looking away frequently (e.g., reading from a screen), it may indicate they are reading AI-generated responses.
      - Gaze percentage (looking away): {gaze_percentage:.2f}% (higher percentage suggests reading behavior).

    Context: {context}.
    Filler words detected: {filler_count}.
    Transcript: "{transcription}"

    Provide:
    - Classification: "Real (Human-Created)" or "Fake (AI-Generated)"
    - Probability Score: 
      - If "Real (Human-Created)": "Real (Human-Created) - <XX>%" (confidence it is human-created)
      - If "Fake (AI-Generated)": "Fake (AI-Generated) - <XX>%" (confidence it is AI-generated)
    - Justification: <Brief Reason, combining linguistic and behavioral analysis>
    """
    response = model.generate_content(prompt)
    return response.text

# -------------------- API Endpoints --------------------
@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Endpoint to analyze an uploaded audio or video file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_type = request.form.get('file_type', 'audio')  # Default to audio
    context = request.form.get('context', 'Formal Interview')
    threshold = float(request.form.get('threshold', 50.0))

    if file_type not in ["audio", "video"]:
        return jsonify({"error": "Invalid file type"}), 400

    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file_type}_{int(time.time())}.{'wav' if file_type == 'audio' else 'mp4'}"
    file.save(temp_file_path)

    # Process the file
    transcription = ""
    gaze_percentage = 0.0

    if file_type == "audio":
        with open(temp_file_path, "rb") as f:
            audio_bytes = f.read()
        transcription = transcribe_audio(audio_bytes)
    else:
        audio_bytes = extract_audio_from_video(temp_file_path)
        if audio_bytes:
            transcription = transcribe_audio(audio_bytes)
        gaze_percentage = analyze_gaze(temp_file_path)

    os.remove(temp_file_path)

    if not transcription or "❌" in transcription:
        return jsonify({"error": "Failed to transcribe audio"}), 500

    # Analyze with Gemini
    analysis = get_gemini_response(transcription, gaze_percentage, context)

    # Parse the response
    try:
        lines = [line.strip() for line in analysis.split("\n") if line.strip()]
        class_line = next(line for line in lines if "Classification:" in line)
        prob_line = next(line for line in lines if "Probability Score:" in line)
        just_line = "\n".join([line for line in lines if "Justification:" in line or line.startswith("-")])

        initial_class = re.search(r"Classification:\s*(.+)", class_line).group(1).strip()
        prob_match = re.search(r"(\d+)%", prob_line)
        if not prob_match:
            raise ValueError("Probability Score not found")

        confidence = int(prob_match.group(1))
        if "Real (Human-Created)" in initial_class:
            human_prob = confidence
            ai_prob = 100 - human_prob
        else:
            ai_prob = confidence
            human_prob = 100 - ai_prob

        final_class = "Fake (AI-Generated)" if ai_prob > threshold else "Real (Human-Created)"
    except (StopIteration, AttributeError, ValueError) as e:
        return jsonify({"error": f"Error parsing analysis: {e}"}), 500

    # Save to database
    conn = init_db()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO results (transcription, classification, human_prob, ai_prob, justification, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (transcription, final_class, human_prob, ai_prob, just_line, time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        conn.close()

    # Return the result
    return jsonify({
        "transcription": transcription,
        "gaze_percentage": gaze_percentage,
        "classification": final_class,
        "human_prob": human_prob,
        "ai_prob": ai_prob,
        "justification": just_line
    })

@app.route('/results', methods=['GET'])
def get_results():
    """Endpoint to retrieve saved analysis results."""
    conn = init_db()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM results ORDER BY timestamp DESC")
        rows = c.fetchall()
        results = [
            {
                "id": row[0],
                "transcription": row[1],
                "classification": row[2],
                "human_prob": row[3],
                "ai_prob": row[4],
                "justification": row[5],
                "timestamp": row[6]
            }
            for row in rows
        ]
        return jsonify(results)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

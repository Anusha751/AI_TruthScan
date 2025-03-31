from flask import Flask, request, jsonify
import google.generativeai as genai
import speech_recognition as sr
import re
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
import os
import sqlite3
import ffmpeg
import time

app = Flask(__name__)

# -------------------- CONFIGURATION --------------------
GOOGLE_API_KEY = "api_key"  # Replace with your actual Google API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# MediaPipe setup for gaze tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Database setup
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
    filler_words = ["um", "uh", "like", "you know", "er", "well"]
    pattern = r'\b(' + '|'.join(filler_words) + r')\b'
    return len(re.findall(pattern, transcription.lower(), re.IGNORECASE))

def extract_audio_from_video(video_bytes):
    output_audio_path = "temp_audio.wav"
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_bytes)
    try:
        ffmpeg.input(temp_video_path).output(output_audio_path, format='wav').run(overwrite_output=True, quiet=True)
        with open(output_audio_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(output_audio_path)
        os.remove(temp_video_path)
        return audio_bytes
    except Exception as e:
        return None

def transcribe_audio(audio_bytes):
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

def analyze_gaze(video_bytes):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(temp_video_path)
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
                left_iris = face_landmarks.landmark[468]
                right_iris = face_landmarks.landmark[473]
                nose_tip = face_landmarks.landmark[1]
                left_gaze_offset = abs(left_iris.x - nose_tip.x)
                right_gaze_offset = abs(right_iris.x - nose_tip.x)
                if left_gaze_offset > 0.1 or right_gaze_offset > 0.1:
                    looking_away_count += 1
    cap.release()
    os.remove(temp_video_path)
    looking_away_percentage = (looking_away_count / total_frames) * 100 if total_frames > 0 else 0
    return looking_away_percentage

def get_gemini_response(transcription, gaze_percentage, context="General"):
    filler_count = count_filler_words(transcription)
    prompt = f"""
    You are an advanced AI content analyzer designed to detect AI-generated responses in spoken text during interviews.
    Your goal is to differentiate between human-spoken responses and AI-generated ones.

    Consider the following factors while analyzing:
    - Linguistic Analysis:
      - Human traits: filler words (e.g., "um," "uh"), self-corrections, personal anecdotes, emotional tone, slight grammatical errors.
      - AI traits: overly formal tone, perfect grammar, repetitive structure, lack of personal depth, generic phrasing.
    - Behavioral Analysis:
      - Gaze tracking: Gaze percentage (looking away): {gaze_percentage:.2f}% (higher percentage suggests reading behavior).

    Context: {context}
    Filler words detected: {filler_count}
    Transcript: "{transcription}"

    Provide:
    - Classification: "Real (Human-Created)" or "Fake (AI-Generated)"
    - Probability Score: <XX>%
    - Justification: <Brief Reason>
    """
    response = model.generate_content(prompt)
    return response.text

# -------------------- API ENDPOINTS --------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    audio_bytes = data.get('audio_bytes', None)
    video_bytes = data.get('video_bytes', None)
    context = data.get('context', 'General')
    threshold = data.get('threshold', 50)

    transcription = ""
    gaze_percentage = 0.0

    if audio_bytes:
        audio_bytes = bytes.fromhex(audio_bytes)
        transcription = transcribe_audio(audio_bytes)
    elif video_bytes:
        video_bytes = bytes.fromhex(video_bytes)
        audio_bytes = extract_audio_from_video(video_bytes)
        if audio_bytes:
            transcription = transcribe_audio(audio_bytes)
        gaze_percentage = analyze_gaze(video_bytes)

    if not transcription:
        return jsonify({"error": "No transcription available"}), 400

    analysis = get_gemini_response(transcription, gaze_percentage, context)
    lines = [line.strip() for line in analysis.split("\n") if line.strip()]
    class_line = next(line for line in lines if "Classification:" in line)
    prob_line = next(line for line in lines if "Probability Score:" in line)
    just_line = "\n".join([line for line in lines if "Justification:" in line or line.startswith("-")])

    initial_class = re.search(r"Classification:\s*(.+)", class_line).group(1).strip()
    prob_match = re.search(r"(\d+)%", prob_line)
    confidence = int(prob_match.group(1)) if prob_match else 50
    human_prob = confidence if "Real" in initial_class else 100 - confidence
    ai_prob = 100 - human_prob
    final_class = "Fake (AI-Generated)" if ai_prob > threshold else "Real (Human-Created)"

    # Save to database
    conn = init_db()
    c = conn.cursor()
    c.execute("INSERT INTO results (transcription, classification, human_prob, ai_prob, justification, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
              (transcription, final_class, human_prob, ai_prob, just_line, time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

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
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM results ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify([{
        "id": row[0],
        "transcription": row[1],
        "classification": row[2],
        "human_prob": row[3],
        "ai_prob": row[4],
        "justification": row[5],
        "timestamp": row[6]
    } for row in rows])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

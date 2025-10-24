import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
import av

# ------------------ STREAMLIT PAGE SETUP ------------------
st.set_page_config(page_title="🎥 Face Recognition Attendance", layout="wide")
st.title("🎯 Real-Time Face Recognition Attendance System")
st.markdown("This app uses your **webcam** to detect and recognize faces, marking attendance automatically.")

# ------------------ DATA SETUP ------------------
if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv("attendance.csv", index=False)

def mark_attendance(name):
    """Add name and timestamp to attendance.csv"""
    df = pd.read_csv("attendance.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name not in df["Name"].values:  # Avoid duplicate entries
        new_entry = pd.DataFrame([[name, now]], columns=["Name", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv("attendance.csv", index=False)

# ------------------ VIDEO PROCESSOR ------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_detected = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]

            try:
                result = DeepFace.find(face_img, db_path="faces_db", enforce_detection=False, silent=True)
                if len(result) > 0 and len(result[0]) > 0:
                    name = os.path.basename(result[0]["identity"][0]).split(".")[0]
                    cv2.putText(img, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    mark_attendance(name)
                else:
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception:
                cv2.putText(img, "Processing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ STREAMLIT WEBCAM ------------------
webrtc_streamer(
    key="face-recognition",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ------------------ SHOW ATTENDANCE TABLE ------------------
st.subheader("📋 Attendance Log")
if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv")
    st.dataframe(df, use_container_width=True)
else:
    st.info("No attendance data yet.")

st.markdown("🧠 Tip: Add known faces in a folder named **faces_db/** before running this app.")
# app.py
# Smart Face Attendance — Stable Build (Improved Face Detection)
# Features:
# - Guided 5-angle registration with hybrid detection (OpenCV + DeepFace)
# - Auto brightness/contrast normalization
# - Single start/stop camera control
# - One attendance per user per day
# - Visual ✅/❌ overlay
# - Compact attendance table

import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import time
import pandas as pd

# Try import DeepFace
try:
    from deepface import DeepFace
except Exception as e:
    st.error("DeepFace not available. Install it: pip install deepface")
    st.stop()

# -------------------------
# Config / Folders
# -------------------------
st.set_page_config(page_title="Smart Face Attendance", layout="wide")
st.title("🎯 Smart Face Recognition Attendance (Improved Build)")

DB_PATH = "attendance.db"
FACES_DIR = "faces_db"
os.makedirs(FACES_DIR, exist_ok=True)

# -------------------------
# Database functions
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT PRIMARY KEY
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            name TEXT,
            day TEXT,
            time TEXT,
            status TEXT,
            UNIQUE(name, day)
        )
    """)
    conn.commit()
    conn.close()

def add_user_db(name):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def mark_attendance_db(name, status):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    now = datetime.now()
    day = now.strftime("%A")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT OR REPLACE INTO attendance (name, day, time, status)
        VALUES (?, ?, ?, ?)
    """, (name, day, time_str, status))
    conn.commit()
    conn.close()

def read_attendance_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, day, time, status FROM attendance ORDER BY time DESC", conn)
    conn.close()
    return df

# -------------------------
# Face detection helpers
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face(frame_rgb):
    """
    Checks if a face exists using OpenCV first, then DeepFace fallback.
    """
    try:
        frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=25)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(70, 70))
        if len(faces) > 0:
            return True
        detections = DeepFace.extract_faces(img_path=frame_rgb, enforce_detection=False)
        if len(detections) > 0:
            return True
    except Exception:
        pass
    return False

def find_identity(frame_rgb, db_path=FACES_DIR):
    """
    Returns recognized name or None
    """
    try:
        res = DeepFace.find(img_path=frame_rgb, db_path=db_path, model_name="Facenet", enforce_detection=False, silent=True)
        if isinstance(res, list) and len(res) > 0 and len(res[0]) > 0:
            identity_path = res[0]['identity'][0]
            name = os.path.basename(identity_path).split("_")[0]
            return name
        if hasattr(res, 'shape') and res.shape[0] > 0:
            identity_path = res.iloc[0]['identity']
            return os.path.basename(identity_path).split("_")[0]
    except Exception:
        pass
    return None

# -------------------------
# Init
# -------------------------
init_db()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "last_marked_today" not in st.session_state:
    st.session_state.last_marked_today = {}

def load_marked_today():
    df = read_attendance_df()
    today_day = datetime.now().strftime("%A")
    today_names = set(df[df["day"] == today_day]["name"].tolist())
    st.session_state.last_marked_today = {n: True for n in today_names}

load_marked_today()

# -------------------------
# UI Tabs
# -------------------------
tab_register, tab_attend, tab_week, tab_admin = st.tabs(["🧍 Register", "🎥 Attendance", "📅 Weekly Summary", "⚙️ Admin"])

# --- Registration ---
with tab_register:
    st.header("Register New User (Guided Multi-Angle)")
    reg_name = st.text_input("Full name to register", key="reg_name")
    if st.button("Start Registration", key="start_registration"):
        if not reg_name.strip():
            st.error("Please enter a valid name.")
        else:
            name = reg_name.strip().replace(" ", "_")
            st.info(f"Registering {name}...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            angles = [
                ("front", "Look straight at the camera"),
                ("left", "Turn your face LEFT"),
                ("right", "Turn your face RIGHT"),
                ("up", "Tilt your face UP"),
                ("down", "Tilt your face DOWN")
            ]
            preview = st.empty()
            success_all = True

            for tag, instruct in angles:
                preview.info(instruct)
                for i in range(3, 0, -1):
                    preview.warning(f"{instruct} — capturing in {i}...")
                    time.sleep(1)

                ret, frame = cap.read()
                if not ret:
                    preview.error("Camera error.")
                    success_all = False
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=25)
                preview.image(frame_rgb, caption=f"Captured {tag}", use_column_width=True)

                if not has_face(frame_rgb):
                    preview.warning(f"No face detected for {tag}. Try adjusting lighting or face angle.")
                    success_all = False
                    break

                save_path = os.path.join(FACES_DIR, f"{name}_{tag}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                time.sleep(0.5)

            cap.release()
            preview.empty()

            if success_all:
                add_user_db(name)
                st.success(f"✅ Registration complete for {name}")
                load_marked_today()
            else:
                st.error("Registration failed. Please retry with better lighting.")

# --- Attendance ---
with tab_attend:
    st.header("Live Attendance (Single Camera Control)")
    col1, col2 = st.columns(2)
    start_clicked = col1.button("▶️ Start Camera", key="start_cam")
    stop_clicked = col2.button("⛔ Stop Camera", key="stop_cam")

    if start_clicked:
        st.session_state.camera_running = True
    if stop_clicked:
        st.session_state.camera_running = False

    frame_display = st.empty()
    status_display = st.empty()

    if st.session_state.camera_running:
        status_display.info("Camera running — align face for recognition...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    status_display.error("Camera read failed.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                faces = face_cascade.detectMultiScale(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), 1.1, 4)
                name_found = None

                if len(faces) > 0:
                    name_found = find_identity(frame_rgb)
                    if name_found:
                        if not st.session_state.last_marked_today.get(name_found):
                            mark_attendance_db(name_found, "Present")
                            st.session_state.last_marked_today[name_found] = True
                            status_display.success(f"✅ Marked attendance for {name_found}")
                        else:
                            status_display.info(f"✅ Already marked today: {name_found}")
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 200, 0), 3)
                            cv2.putText(frame_rgb, f"{name_found} ✓", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
                    else:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (200, 0, 0), 3)
                            cv2.putText(frame_rgb, "Unknown ✗", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 3)
                        status_display.error("❌ Face not recognized")
                else:
                    status_display.info("No face detected — please align properly")

                frame_display.image(frame_rgb, use_column_width=True)
                time.sleep(0.07)

                if not st.session_state.camera_running:
                    break
        finally:
            cap.release()
            frame_display.empty()
            status_display.info("Camera stopped.")
    else:
        frame_display.info("Camera is stopped. Click ▶️ Start Camera to begin.")

# --- Weekly Summary ---
with tab_week:
    st.header("Weekly Summary (Mon–Fri)")
    df = read_attendance_df()
    if df.empty:
        st.info("No attendance records yet.")
    else:
        users = sorted(pd.read_sql_query("SELECT name FROM users", sqlite3.connect(DB_PATH))["name"].tolist())
        if not users:
            st.info("No registered users yet.")
        else:
            person = st.selectbox("Select person", users)
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            summary = []
            user_df = df[df["name"] == person]
            for d in days:
                present = any(user_df["day"] == d)
                summary.append({"Day": d, "Status": "✅" if present else "❌"})
            st.table(pd.DataFrame(summary))

# --- Admin ---
with tab_admin:
    st.header("Admin")
    st.warning("This will delete all data and face images.")
    if st.button("🗑️ Delete ALL records & faces", key="delete_all_confirm"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM users")
        c.execute("DELETE FROM attendance")
        conn.commit()
        conn.close()
        for f in os.listdir(FACES_DIR):
            try:
                os.remove(os.path.join(FACES_DIR, f))
            except Exception:
                pass
        st.session_state.last_marked_today = {}
        st.success("All data and faces deleted successfully.")

# --- Today's Attendance Footer ---
st.markdown("---")
st.subheader("Today's Attendance (Latest per Person)")
df_today = read_attendance_df()
today = datetime.now().strftime("%A")
df_today = df_today[df_today["day"] == today]
if not df_today.empty:
    df_compact = df_today.drop_duplicates(subset=["name"], keep="last")[["name", "time", "status"]]
    st.dataframe(df_compact, use_container_width=True, height=250)
else:
    st.info("No attendance recorded today yet.")

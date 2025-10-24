# app.py
# Single-file Streamlit app: robust multi-angle registration + single-camera live recognition
# Features:
# - Guided 5-angle registration (front/left/right/up/down) with countdown & face-check
# - Single Start/Stop camera controls (no duplicate buttons)
# - Safe SQLite access (one record per person per day, UNIQUE constraint)
# - Fast recognition loop that marks attendance once per person per day
# - Visual ✅/❌ overlay on live feed
# - Compact attendance table (no long scrolling)
#
# Requirements:
# pip install streamlit opencv-python-headless deepface pandas numpy

import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import time
import pandas as pd

# try import DeepFace (fail gracefully)
try:
    from deepface import DeepFace
except Exception as e:
    st.error("DeepFace not available. Install deepface and required backends. Error: " + str(e))
    st.stop()

# -------------------------
# Config / ensure folders
# -------------------------
st.set_page_config(page_title="Smart Face Attendance", layout="wide")
st.title("🎯 Smart Face Recognition Attendance (Stable Build)")

DB_PATH = "attendance.db"
FACES_DIR = "faces_db"
os.makedirs(FACES_DIR, exist_ok=True)

# -------------------------
# Database utilities
# -------------------------
def init_db():
    # create DB with appropriate schema and unique constraint (one record per person per day)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL")           # reduce locking problems
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
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
        conn.commit()
    finally:
        conn.close()

def mark_attendance_db(name, status):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        now = datetime.now()
        day = now.strftime("%A")
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        # insert or replace ensures one record per day (keeps latest time)
        c.execute("""
            INSERT OR REPLACE INTO attendance (name, day, time, status)
            VALUES (?, ?, ?, ?)
        """, (name, day, time_str, status))
        conn.commit()
    finally:
        conn.close()

def read_attendance_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, day, time, status FROM attendance ORDER BY time DESC", conn)
    conn.close()
    return df

# -------------------------
# Helpers: face check & embeddings
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def find_identity(frame_rgb, db_path=FACES_DIR, threshold=0.80):
    # return matched name or None
    try:
        # DeepFace.find expects either file path or numpy array; we pass array
        res = DeepFace.find(img_path = frame_rgb, db_path = db_path, model_name="Facenet", enforce_detection=False, silent=True)
        if isinstance(res, list) and len(res) > 0 and len(res[0]) > 0:
            # result[0]['identity'][0] is a path like faces_db/Name_front.jpg
            identity_path = res[0]['identity'][0]
            basename = os.path.basename(identity_path)
            # name before first underscore
            name = basename.split("_")[0]
            return name
        # fallback: DeepFace.find sometimes returns a DataFrame directly
        if hasattr(res, 'shape'):
            # DataFrame-like: check non-empty
            if res.shape[0] > 0:
                identity_path = res.iloc[0]['identity']
                return os.path.basename(identity_path).split("_")[0]
    except Exception:
        # ignore recognition exceptions (model loading, etc.) to keep loop alive
        pass
    return None

# -------------------------
# Initialization
# -------------------------
init_db()

# -------------------------
# Session state for single camera control
# -------------------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "last_marked_today" not in st.session_state:
    # dict to keep track of who was marked today in this session (fast in-memory guard)
    st.session_state.last_marked_today = {}

# Utility: check DB for today's marks to keep in-memory consistent with DB
def load_marked_today():
    df = read_attendance_df()
    today_name_set = set()
    today_day = datetime.now().strftime("%A")
    for _, row in df.iterrows():
        if row["day"] == today_day:
            today_name_set.add(row["name"])
    st.session_state.last_marked_today = {n: True for n in today_name_set}

load_marked_today()

# -------------------------
# UI: Tabs for Register, Attendance, Weekly Summary, Admin
# -------------------------
tab_register, tab_attend, tab_week, tab_admin = st.tabs(["🧍 Register", "🎥 Attendance", "📅 Weekly Summary", "⚙️ Admin"])

# --- Registration tab ---
with tab_register:
    st.header("Register new user (guided multi-angle)")
    reg_name = st.text_input("Full name to register", key="reg_name")
    if "register_feedback" not in st.session_state:
        st.session_state.register_feedback = ""

    if st.button("Start Registration", key="start_registration"):
        if not reg_name or reg_name.strip() == "":
            st.error("Please enter a valid name before registration.")
        else:
            name = reg_name.strip().replace(" ", "_")  # store filenames with underscores
            st.session_state.register_feedback = f"Registering {name}..."
            # open camera and guide
            cap = cv2.VideoCapture(0)
            angles = [("front", "Look straight at camera"),
                      ("left", "Turn your face to the LEFT"),
                      ("right", "Turn your face to the RIGHT"),
                      ("up", "Tilt your face UP"),
                      ("down", "Tilt your face DOWN")]
            preview = st.empty()
            success_all = True
            for tag, instruct in angles:
                preview.info(instruct)
                # countdown 3..1
                for i in range(3, 0, -1):
                    preview.warning(f"{instruct} — capturing in {i}...")
                    time.sleep(1)
                ret, frame = cap.read()
                if not ret:
                    preview.error("Camera read failed. Try again.")
                    success_all = False
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview.image(frame_rgb, caption=f"Captured {tag}", use_column_width=True)
                # check face present
                if not has_face(frame_rgb):
                    preview.error(f"No face detected for {tag}. Please retry registration and ensure good lighting.")
                    success_all = False
                    break
# save as name_tag.jpg (BGR saved)
                save_path = os.path.join(FACES_DIR, f"{name}_{tag}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                time.sleep(0.5)
            cap.release()
            preview.empty()
            if success_all:
                add_user_db(name)
                st.success(f"Registration complete for {name}. You can now sign-in using the Attendance tab.")
                load_marked_today()  # refresh in-memory marks
            else:
                st.error("Registration failed. Please try again (better lighting, stable camera).")

# --- Attendance tab ---
with tab_attend:
    st.header("Live Attendance (single Start/Stop buttons)")

    col1, col2 = st.columns([1,1])
    with col1:
        start_clicked = st.button("▶️ Start Camera", key="start_cam")
    with col2:
        stop_clicked = st.button("⛔ Stop Camera", key="stop_cam")

    # Respect the Start/Stop buttons and session state
    if start_clicked:
        st.session_state.camera_running = True
    if stop_clicked:
        st.session_state.camera_running = False

    frame_display = st.empty()
    status_display = st.empty()

    if st.session_state.camera_running:
        status_display.info("Camera running — aligning face for recognition...")
        cap = cv2.VideoCapture(0)
        # set a slightly higher fps if supported
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            start_time = time.time()
            # run for a long time until user stops
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    status_display.error("Camera read failed.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Optional: show small detection rectangle(s) for visual hint
                try:
                    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), 1.1, 4)
                except Exception:
                    faces = []

                name_found = None
                if len(faces) > 0:
                    # attempt recognition once per visible frame
                    name_found = find_identity(frame_rgb)
                    if name_found:
                        # avoid duplicate marking same day (DB enforced, but quick in-memory guard helps)
                        if not st.session_state.last_marked_today.get(name_found, False):
                            mark_attendance_db(name_found, "Present")
                            st.session_state.last_marked_today[name_found] = True
                            status_display.success(f"✅ Marked attendance for {name_found}")
                        else:
                            status_display.info(f"✅ Already marked today: {name_found}")
                        # draw green box + tick
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 200, 0), 3)
                            cv2.putText(frame_rgb, f"{name_found} ✓", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
                    else:
                        # draw red box + cross for first face
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (200, 0, 0), 3)
                            cv2.putText(frame_rgb, "Unknown ✗", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 3)
                        status_display.error("❌ Face not recognized")
                else:
                    status_display.info("No face detected — please align to the camera")

                frame_display.image(frame_rgb, use_column_width=True)

                # small sleep so UI stays responsive and CPU isn't pegged
                time.sleep(0.07)

                # re-evaluate stop button without creating duplicates
                # If user clicked Stop (setting session state false via earlier button), break
                if not st.session_state.camera_running:
                    break

        finally:
            cap.release()
            frame_display.empty()
            status_display.info("Camera stopped.")
    else:
        frame_display.info("Camera is stopped. Click ▶️ Start Camera to begin.")

# --- Weekly Summary tab (Monday-Friday) ---
with tab_week:
    st.header("Weekly (Mon–Fri) Summary per person")
    df = read_attendance_df()
    if df.empty:
        st.info("No attendance records yet.")
    else:
        # build summary for Mon-Fri for each registered user
        users = sorted(pd.read_sql_query("SELECT name FROM users", sqlite3.connect(DB_PATH))["name"].tolist())
        if not users:
            st.info("No registered users yet.")
        else:
            person = st.selectbox("Select person", users)
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            # prepare a simple row
            today_records = df[df["name"] == person]
            summary = []
            for d in days:
                present = any(today_records["day"] == d)
                summary.append({"Day": d, "Status": "✅" if present else "❌"})
            st.table(pd.DataFrame(summary))

# --- Admin tab ---
with tab_admin:
    st.header("Admin")
    st.markdown("Danger zone — this will remove all users, faces and attendance records.")
    if st.button("🗑️ Delete ALL records & faces (CONFIRM)", key="delete_all_confirm"):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM users")
            c.execute("DELETE FROM attendance")
            conn.commit()
            conn.close()
        except Exception as e:
            st.error("DB delete error: " + str(e))
        # delete face images
        for f in os.listdir(FACES_DIR):
            try:
                os.remove(os.path.join(FACES_DIR, f))
            except Exception:
                pass
        # reset session-state markers
        st.session_state.last_marked_today = {}
        st.success("All data and faces removed.")

# -------------------------
# Compact view of today's attendance (footer)
# -------------------------
st.markdown("---")
st.subheader("Today's attendance (latest per person)")
df_today = read_attendance_df()
if not df_today.empty:
    today = datetime.now().strftime("%A")
    df_today = df_today[df_today["day"] == today]
    # drop duplicates (keep last per name) and show compact
    if not df_today.empty:
        df_compact = df_today.drop_duplicates(subset=["name"], keep="last")[["name", "time", "status"]]
        st.dataframe(df_compact, use_container_width=True, height=250)
    else:
        st.info("No attendance today yet.")
else:
    st.info("No attendance recorded yet.")


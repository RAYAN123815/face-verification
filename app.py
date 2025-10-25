# app.py
# Smart Face Attendance — Hybrid Local & Cloud Compatible
# - Uses OpenCV locally
# - Uses st.camera_input() on Streamlit Cloud
# - Multi-angle registration with retries and lighting check

import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import time
import pandas as pd

# ------------------ Load DeepFace safely ------------------
try:
    from deepface import DeepFace
except Exception as e:
    st.error(f"⚠️ DeepFace not available. Install it: pip install deepface\n\n{e}")
    st.stop()

# ------------------ Config ------------------
st.set_page_config(page_title="Smart Face Attendance", layout="wide")
st.title("🎯 Smart Face Recognition Attendance (Hybrid Compatible)")

DB_PATH = "attendance.db"
FACES_DIR = "faces_db"
os.makedirs(FACES_DIR, exist_ok=True)

# ------------------ Database ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY)")
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
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def mark_attendance_db(name, status="Present"):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now()
    conn.execute(
        "INSERT OR REPLACE INTO attendance VALUES (?, ?, ?, ?)",
        (name, now.strftime("%A"), now.strftime("%Y-%m-%d %H:%M:%S"), status)
    )
    conn.commit()
    conn.close()

def read_attendance_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY time DESC", conn)
    conn.close()
    return df

# ------------------ Face Detection Helpers ------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face(frame_rgb):
    """Detect face using OpenCV or DeepFace fallback"""
    try:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return True
        det = DeepFace.extract_faces(img_path=frame_rgb, enforce_detection=False)
        return len(det) > 0
    except Exception:
        return False

def find_identity(frame_rgb, db_path=FACES_DIR):
    """Find known face identity"""
    try:
        res = DeepFace.find(img_path=frame_rgb, db_path=db_path,
                            model_name="Facenet", enforce_detection=False, silent=True)
        if isinstance(res, list) and len(res) > 0 and len(res[0]) > 0:
            identity_path = res[0]["identity"][0]
            return os.path.basename(identity_path).split("_")[0]
        if hasattr(res, "iloc") and len(res) > 0:
            identity_path = res.iloc[0]["identity"]
            return os.path.basename(identity_path).split("_")[0]
    except Exception:
        pass
    return None

# ------------------ Init ------------------
init_db()
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "last_marked_today" not in st.session_state:
    st.session_state.last_marked_today = {}

def load_marked_today():
    df = read_attendance_df()
    today = datetime.now().strftime("%A")
    today_users = set(df[df["day"] == today]["name"])
    st.session_state.last_marked_today = {u: True for u in today_users}

load_marked_today()

# ------------------ UI Tabs ------------------
tab_register, tab_attend, tab_week, tab_admin = st.tabs([
    "🧍 Register", "🎥 Attendance", "📅 Weekly Summary", "⚙️ Admin"
])

# ------------------ Registration ------------------
with tab_register:
    st.header("Register New User (Hybrid Capture)")
    reg_name = st.text_input("Full name to register", key="reg_name")

    if st.button("Start Registration", key="start_registration"):
        if not reg_name.strip():
            st.error("Please enter a valid name first.")
        else:
            name = reg_name.strip().replace(" ", "_")
            st.info(f"Starting registration for {name}...")

            # Try local camera first
            cap = None
            for idx in (0, 1, 2):
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    cap = test
                    break
                test.release()

            # --- ONLINE fallback ---
            use_streamlit_camera = False
            if not cap or not cap.isOpened():
                st.warning("⚠️ No physical camera found. Using browser camera instead.")
                use_streamlit_camera = True

            angles = [
                ("front", "Look straight at the camera"),
                ("left", "Turn LEFT"),
                ("right", "Turn RIGHT"),
                ("up", "Tilt UP"),
                ("down", "Tilt DOWN")
            ]

            preview = st.empty()
            success_all = True

            for tag, instruction in angles:
                st.subheader(instruction)

                if use_streamlit_camera:
                    # --- Browser camera ---
                    photo = st.camera_input(f"Capture your {tag} face")
                    if photo:
                        frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if has_face(frame_rgb):
                            cv2.imwrite(os.path.join(FACES_DIR, f"{name}_{tag}.jpg"),
                                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                            st.success(f"✅ {tag} captured successfully!")
                        else:
                            st.error("No face detected. Try retaking with better lighting.")
                            success_all = False
                            break
                    else:
                        st.warning("No photo captured.")
                        success_all = False
                        break
                else:
                    # --- Local OpenCV camera ---
                    preview.info(f"{instruction} — hold still.")
                    for i in range(3, 0, -1):
                        preview.warning(f"Capturing in {i}...")
                        time.sleep(1)

                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera read failed.")
                        success_all = False
                        break

                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview.image(frame_rgb, caption=f"Captured {tag}", use_column_width=True)

                    if not has_face(frame_rgb):
                        st.error("No face detected. Please ensure good lighting.")
                        success_all = False
                        break

                    cv2.imwrite(os.path.join(FACES_DIR, f"{name}_{tag}.jpg"),
                                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    time.sleep(0.5)

            if not use_streamlit_camera:
                cap.release()

            if success_all:
                add_user_db(name)
                st.success(f"✅ Registration complete for {name}")
                load_marked_today()
            else:
                st.error("Registration failed. Please retry with better lighting or stable camera.")

# ------------------ Attendance ------------------
with tab_attend:
    st.header("🎥 Live Attendance (Hybrid Mode)")
    st.write("Works locally via webcam or online via browser camera.")

    # Try to open local camera to detect if we're running locally
    cap_test = cv2.VideoCapture(0)
    local_camera = cap_test.isOpened()
    cap_test.release()

    frame_display = st.empty()
    status_display = st.empty()

    if not local_camera:
        # --- CLOUD MODE (Streamlit camera input) ---
        st.info("🌐 Running in cloud mode — using browser camera for attendance.")
        uploaded_image = st.camera_input("Align your face and capture to mark attendance")

        if uploaded_image:
            frame = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not has_face(frame_rgb):
                st.error("❌ No face detected. Please ensure good lighting.")
            else:
                name_found = find_identity(frame_rgb)
                if name_found:
                    if not st.session_state.last_marked_today.get(name_found):
                        mark_attendance_db(name_found)
                        st.session_state.last_marked_today[name_found] = True
                        st.success(f"✅ Attendance marked for {name_found}")
                    else:
                        st.info(f"✅ Already marked today: {name_found}")
                else:
                    st.error("❌ Face not recognized. Try again.")

    else:
        # --- LOCAL MODE (OpenCV live camera) ---
        col1, col2 = st.columns(2)
        start_clicked = col1.button("▶️ Start Camera")
        stop_clicked = col2.button("⛔ Stop Camera")

        if start_clicked:
            st.session_state.camera_running = True
        if stop_clicked:
            st.session_state.camera_running = False

        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            status_display.info("Camera running — align face for recognition...")

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    status_display.error("Camera read failed.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_cascade.detectMultiScale(
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), 1.1, 4
                )

                if len(faces) > 0:
                    name_found = find_identity(frame_rgb)
                    if name_found:
                        if not st.session_state.last_marked_today.get(name_found):
                            mark_attendance_db(name_found)
                            st.session_state.last_marked_today[name_found] = True
                            status_display.success(f"✅ Attendance marked for {name_found}")
                        else:
                            status_display.info(f"✅ Already marked today: {name_found}")
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 200, 0), 3)
                            cv2.putText(frame_rgb, f"{name_found} ✓", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
                    else:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (200, 0, 0), 3)
                            cv2.putText(frame_rgb, "Unknown ✗", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 3)
                        status_display.error("❌ Face not recognized")
                else:
                    status_display.info("No face detected — align properly")

                frame_display.image(frame_rgb, use_column_width=True)
                time.sleep(0.07)

            cap.release()
            frame_display.empty()
            status_display.info("Camera stopped.")
        else:
            frame_display.info("Camera stopped. Click ▶️ Start Camera to begin.")


# ------------------ Weekly Summary ------------------
with tab_week:
    st.header("Weekly Summary (Mon–Fri)")
    df = read_attendance_df()
    if df.empty:
        st.info("No attendance yet.")
    else:
        users = pd.read_sql_query("SELECT name FROM users", sqlite3.connect(DB_PATH))["name"].tolist()
        if not users:
            st.info("No registered users yet.")
        else:
            person = st.selectbox("Select person", users)
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            user_df = df[df["name"] == person]
            summary = [{"Day": d, "Status": "✅" if any(user_df["day"] == d) else "❌"} for d in days]
            st.table(pd.DataFrame(summary))

# ------------------ Admin ------------------
with tab_admin:
    st.header("Admin")
    st.warning("This will delete all data and images.")
    if st.button("🗑️ Delete ALL"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM attendance")
        conn.commit()
        conn.close()
        for f in os.listdir(FACES_DIR):
            try:
                os.remove(os.path.join(FACES_DIR, f))
            except Exception:
                pass
        st.session_state.last_marked_today = {}
        st.success("All data and faces deleted.")

# ------------------ Footer ------------------
st.markdown("---")
st.subheader("📅 Today's Attendance")
df_today = read_attendance_df()
today = datetime.now().strftime("%A")
df_today = df_today[df_today["day"] == today]
if not df_today.empty:
    st.dataframe(df_today.drop_duplicates("name", keep="last")[["name", "time", "status"]],
                 use_container_width=True, height=250)
else:
    st.info("No attendance recorded today yet.")

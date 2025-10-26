# app.py
# Smart Face Attendance — Hybrid Local & Cloud Compatible
# - Uses OpenCV locally
# - Uses st.camera_input() on Streamlit Cloud
# - Multi-angle registration with instant preview and smooth feedback

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

            cap = None
            for idx in (0, 1, 2):
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    cap = test
                    break
                test.release()

            use_streamlit_camera = False
            if not cap or not cap.isOpened():
                st.warning("⚠️ No physical camera found. Using browser camera instead.")
                use_streamlit_camera = True

            angles = [
                ("front", "Look straight at the camera"),
                ("left", "Turn LEFT"),
                ("right", "Turn RIGHT")
            ]

            success_all = True
            preview = st.empty()

            for tag, instruction in angles:
                st.subheader(instruction)

                if use_streamlit_camera:
                    photo = st.camera_input(f"Capture your {tag} face")
                    if photo:
                        frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if has_face(frame_rgb):
                            img_path = os.path.join(FACES_DIR, f"{name}_{tag}.jpg")
                            cv2.imwrite(img_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                            st.image(frame_rgb, caption=f"{tag.capitalize()} captured successfully!", use_container_width=True)
                        else:
                            st.error("No face detected. Try again.")
                            success_all = False
                            break
                    else:
                        st.warning("No photo captured.")
                        success_all = False
                        break
                else:
                    preview.info(f"{instruction} — hold still.")
                    time.sleep(2)
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera read failed.")
                        success_all = False
                        break
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not has_face(frame_rgb):
                        st.error("No face detected. Please ensure good lighting.")
                        success_all = False
                        break
                    img_path = os.path.join(FACES_DIR, f"{name}_{tag}.jpg")
                    cv2.imwrite(img_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    st.image(frame_rgb, caption=f"{tag.capitalize()} captured!", use_container_width=True)

            if not use_streamlit_camera and cap:
                cap.release()

            if success_all:
                add_user_db(name)
                st.success(f"✅ {name} registered successfully!")
                st.balloons()
                load_marked_today()
            else:
                st.error("Registration failed. Try again with better lighting.")

# ------------------ Attendance ------------------
with tab_attend:
    st.header("🎥 Mark Attendance")
    st.write("Works via webcam (local) or browser camera (cloud).")

    cap_test = cv2.VideoCapture(0)
    local_camera = cap_test.isOpened()
    cap_test.release()

    if not local_camera:
        st.info("🌐 Using browser camera.")
        uploaded_image = st.camera_input("Capture your face to mark attendance")

        if uploaded_image:
            frame = cv2.imdecode(np.frombuffer(uploaded_image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not has_face(frame_rgb):
                st.error("❌ No face detected.")
            else:
                name_found = find_identity(frame_rgb)
                if name_found:
                    if not st.session_state.last_marked_today.get(name_found):
                        mark_attendance_db(name_found)
                        st.session_state.last_marked_today[name_found] = True
                        st.success(f"✅ Attendance marked for {name_found}")
                        st.image(frame_rgb, caption=f"{name_found} marked present ✅", use_container_width=True)
                    else:
                        st.info(f"✅ Already marked today: {name_found}")
                else:
                    st.error("❌ Face not recognized. Try again.")
    else:
        st.warning("For better compatibility, use Streamlit camera when running online.")

# ------------------ Weekly Summary ------------------
with tab_week:
    st.header("📅 Weekly Summary")
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
    st.warning("⚠️ This will delete all data and images.")
    if st.button("🗑️ Delete ALL"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM attendance")
        conn.commit()
        conn.close()
        for f in os.listdir(FACES_DIR):
            os.remove(os.path.join(FACES_DIR, f))
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
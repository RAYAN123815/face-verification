# smooth_app.py
# Optimized Face Attendance App — smooth camera + instant recognition

import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
import pandas as pd
import time
from datetime import datetime
from deepface import DeepFace

# ------------------ CONFIG ------------------
DB_PATH = "attendance.db"
FACES_DIR = "faces_db"
os.makedirs(FACES_DIR, exist_ok=True)

st.set_page_config(page_title="Smart Face Attendance", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom right, #e0f7fa, #e1bee7);
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align:center;
        font-size:40px;
        color:#4A148C;
        margin-bottom:10px;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        color:#6A1B9A;
        margin-bottom:25px;
    }
    .camera-box {
        border:3px solid #6A1B9A;
        border-radius:15px;
        overflow:hidden;
        box-shadow:0px 4px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------ DATABASE ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY)")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
        name TEXT,
        day TEXT,
        time TEXT,
        status TEXT,
        UNIQUE(name, day)
    )""")
    conn.commit()
    conn.close()

def add_user(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users VALUES (?)", (name,))
    conn.commit()
    conn.close()

def mark_attendance(name, status="Present"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    day = now.strftime("%A")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT OR REPLACE INTO attendance (name, day, time, status) VALUES (?, ?, ?, ?)",
        (name, day, time_str, status),
    )
    conn.commit()
    conn.close()

def read_attendance():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM attendance ORDER BY time DESC", conn)
    conn.close()
    return df

# ------------------ EMBEDDINGS ------------------
def get_embeddings():
    embeds = {}
    for img_name in os.listdir(FACES_DIR):
        path = os.path.join(FACES_DIR, img_name)
        try:
            rep = DeepFace.represent(img_path=path, model_name="Facenet", enforce_detection=False)
            emb = np.array(rep[0]["embedding"])
            name = img_name.split("_")[0]
            embeds.setdefault(name, []).append(emb)
        except:
            pass
    return embeds

EMBED_CACHE = get_embeddings()

def add_face(name, frame_rgb):
    filename = f"{name}_{int(time.time())}.jpg"
    path = os.path.join(FACES_DIR, filename)
    cv2.imwrite(path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    add_user(name)
    rep = DeepFace.represent(img_path=path, model_name="Facenet", enforce_detection=False)
    emb = np.array(rep[0]["embedding"])
    EMBED_CACHE.setdefault(name, []).append(emb)

def recognize(frame_rgb):
    try:
        rep = DeepFace.represent(frame_rgb, model_name="Facenet", enforce_detection=False)
        emb = np.array(rep[0]["embedding"])
    except:
        return None
    min_dist = 0.8
    best = None
    for name, embs in EMBED_CACHE.items():
        for e in embs:
            dist = np.linalg.norm(emb - e)
            if dist < min_dist:
                min_dist = dist
                best = name
    return best

# ------------------ UI ------------------
st.markdown('<div class="title">🎥 Smart Face Attendance</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smooth Camera | Instant Recognition | Automatic Attendance</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Register", "Attendance"])

# ---------- Register -----------
with tab1:
    name = st.text_input("Enter your name to register")
    if st.button("Register Face"):
        if not name:
            st.error("Please enter your name first!")
        else:
            cap = cv2.VideoCapture(0)
            st.info("Capturing your face — look straight at the camera.")
            preview = st.empty()
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Capturing in {i}")
                time.sleep(1)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                add_face(name, frame_rgb)
                st.success(f"✅ {name} registered successfully!")
            else:
                st.error("Camera capture failed. Try again!")

# ---------- Attendance -----------
with tab2:
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Camera"):
        st.session_state.camera_on = True
    if col2.button("⛔ Stop Camera"):
        st.session_state.camera_on = False

    cam_placeholder = st.empty()
    status = st.empty()

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_mark = {}
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_small = cv2.resize(frame, (300, 300))  # smaller = faster
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            name = recognize(frame_rgb)
            label = "Unknown"
            color = (0, 0, 255)
            if name:
                label = f"{name} ✓"
                color = (0, 255, 0)
                today = datetime.now().strftime("%A")
                if name not in last_mark or last_mark[name] != today:
                    mark_attendance(name)
                    last_mark[name] = today
                    status.success(f"✅ Attendance marked for {name}")
            else:
                status.warning("❌ Face not recognized")

            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            time.sleep(0.08)
        cap.release()
        status.info("Camera stopped.")
    else:
        st.info("Click ▶️ Start Camera to begin recognition.")

# ---------- Today's Attendance ----------
st.markdown("---")
st.subheader("📅 Today's Attendance")
df = read_attendance()
today = datetime.now().strftime("%A")
if df.empty:
    st.info("No attendance yet.")
else:
    df_today = df[df["day"] == today].drop_duplicates(subset=["name"], keep="last")
    st.dataframe(df_today, use_container_width=True)

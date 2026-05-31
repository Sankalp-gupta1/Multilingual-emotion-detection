# =========================================================
# FINAL INDUSTRY STYLE app.py
# MULTI CLIENT + SHARED VIDEO + EMOTION SYSTEM
# =========================================================

import json
import os
import time
import cv2
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from collections import deque

from webcam import start_camera, stop_camera, detect_emotion

from camera_manager import (
    request_camera,
    release_camera,
    get_camera_status
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Distributed AI Emotion System",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #12203a 0%, #080d18 35%, #05070d 100%);
    color: white;
}

.block-container {
    padding-top: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
    border-right: 1px solid #1e293b;
}

.hero-box {
    padding: 28px;
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(0,255,213,0.16), rgba(37,99,235,0.15));
    border: 1px solid rgba(0,255,213,0.35);
    box-shadow: 0 0 35px rgba(0,255,213,0.12);
    margin-bottom: 25px;
}

.main-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #00ffd5, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
}

.sub-title {
    color: #cbd5e1;
    font-size: 17px;
    margin-top: 8px;
}

.glass-card {
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}

.chat-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 15px;
    border-left: 5px solid #00ffd5;
    box-shadow: 0 8px 20px rgba(0,0,0,0.28);
}

.chat-user {
    font-size: 15px;
    color: #7dd3fc;
    font-weight: 700;
}

.chat-text {
    font-size: 16px;
    color: #f8fafc;
    margin-top: 12px;
    line-height: 1.6;
}

.result-pill {
    display: inline-block;
    padding: 8px 13px;
    margin-top: 10px;
    margin-right: 8px;
    border-radius: 999px;
    background: rgba(0,255,213,0.12);
    border: 1px solid rgba(0,255,213,0.35);
    color: #e0f2fe;
    font-weight: 600;
}

.emotion-box {
    background: rgba(2, 6, 23, 0.85);
    padding: 13px;
    border-radius: 14px;
    margin-bottom: 10px;
    border: 1px solid #334155;
}

.status-card {
    padding: 14px;
    border-radius: 16px;
    background: rgba(34,197,94,0.13);
    border: 1px solid rgba(34,197,94,0.3);
    margin-bottom: 12px;
}

.warning-card {
    padding: 14px;
    border-radius: 16px;
    background: rgba(239,68,68,0.13);
    border: 1px solid rgba(239,68,68,0.3);
    margin-bottom: 12px;
}

.metric-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
}

.robot-note {
    color: #94a3b8;
    font-size: 13px;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# ROBOT MASCOT COMPONENT
# =========================================================

components.html("""
<div id="botBox">
    <div id="robot">
        <div class="antenna"></div>
        <div class="head">
            <div class="eye left"></div>
            <div class="eye right"></div>
            <div class="mouth"></div>
        </div>
        <div class="body">AI</div>
        <div class="hand">👉</div>
    </div>
    <div class="msg">I am watching your Activity 👀</div>
</div>

<style>
#botBox {
    height: 135px;
    width: 100%;
    position: relative;
    overflow: hidden;
    background: linear-gradient(90deg, #020617, #0f172a);
    border: 1px solid rgba(0,255,213,0.3);
    border-radius: 22px;
    box-shadow: 0 0 25px rgba(0,255,213,0.12);
}

#robot {
    position: absolute;
    left: 30px;
    top: 18px;
    transition: all 0.15s ease;
}

.antenna {
    width: 4px;
    height: 18px;
    background: #00ffd5;
    margin-left: 42px;
    border-radius: 8px;
}

.head {
    width: 90px;
    height: 58px;
    background: linear-gradient(135deg, #38bdf8, #8b5cf6);
    border-radius: 20px;
    position: relative;
    box-shadow: 0 0 20px rgba(56,189,248,0.5);
}

.eye {
    width: 13px;
    height: 13px;
    background: #020617;
    border-radius: 50%;
    position: absolute;
    top: 18px;
}

.left { left: 22px; }
.right { right: 22px; }

.mouth {
    width: 32px;
    height: 6px;
    background: #020617;
    border-radius: 10px;
    position: absolute;
    left: 29px;
    bottom: 13px;
}

.body {
    width: 70px;
    height: 35px;
    background: #00ffd5;
    color: #020617;
    font-weight: 900;
    text-align: center;
    line-height: 35px;
    margin-left: 10px;
    border-radius: 14px;
    margin-top: 5px;
}

.hand {
    position: absolute;
    left: 88px;
    top: 55px;
    font-size: 30px;
    transform-origin: left center;
    animation: wave 1s infinite alternate;
}

.msg {
    position: absolute;
    left: 160px;
    top: 48px;
    font-family: Segoe UI;
    font-weight: 700;
    color: #e0f2fe;
    font-size: 20px;
}

@keyframes wave {
    from { transform: rotate(-10deg); }
    to { transform: rotate(20deg); }
}
</style>

<script>
const box = document.getElementById("botBox");
const robot = document.getElementById("robot");

box.addEventListener("mousemove", (e) => {
    const rect = box.getBoundingClientRect();
    let x = e.clientX - rect.left - 55;
    if (x < 20) x = 20;
    if (x > rect.width - 140) x = rect.width - 140;
    robot.style.left = x + "px";
});
</script>
""", height=150)

# =========================================================
# FILES
# =========================================================

CHAT_FILE = "global_chat.json"
USERS_FILE = "online_users.json"

if not os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# =========================================================
# FUNCTIONS
# =========================================================

def load_messages():
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_message(msg):
    messages = load_messages()
    messages.append(msg)
    messages = messages[-100:]
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

def load_users():

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

            if isinstance(data, dict):
                return data

            return {}

    except:
        return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

# =========================================================
# SESSION STATE
# =========================================================

if "video_active" not in st.session_state:
    st.session_state.video_active = False

if "camera_user" not in st.session_state:
    st.session_state.camera_user = None

if "video_emotions" not in st.session_state:
    st.session_state.video_emotions = deque(maxlen=10)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero-box">
    <div class="main-title">🔥 Distributed AI Emotion Detection System</div>
    <div class="sub-title">
        Live Multi-Client Chat • Facial Emotion Recognition • Sarcasm Detection • Real-Time Analytics
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🤖 User Control Panel")

username = st.sidebar.text_input("Enter Username", value="Sankalp")

st.sidebar.markdown(
    f"<div class='status-card'>✅ Connected : <b>{username}</b></div>",
    unsafe_allow_html=True
)

users = load_users()

# =========================================================
# FIX OLD FORMAT
# =========================================================

if not isinstance(users, dict):
    users = {}

# =========================================================
# ADD / UPDATE CURRENT USER
# =========================================================

users[username] = {
    "last_seen": time.time()
}

save_users(users)

status = get_camera_status()

if status["active"]:
    st.sidebar.markdown(
        f"<div class='warning-card'>🎥 Camera Active : <b>{status['user']}</b></div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        "<div class='status-card'>🎥 Camera Available</div>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("## 🟢 Online Users")

for user in users.keys():

    if status["active"] and status["user"] == user:
        st.sidebar.markdown(f"🎥 **{user}**  <span style='color:#22c55e'>(VIDEO ON)</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"📴 **{user}**  <span style='color:#ef4444'>(VIDEO OFF)</span>", unsafe_allow_html=True)

# =========================================================
# TOP METRICS
# =========================================================

messages = load_messages()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"<div class='metric-card'><h3>{len(users)}</h3><p>Online Users</p></div>", unsafe_allow_html=True)

with m2:
    st.markdown(f"<div class='metric-card'><h3>{len(messages)}</h3><p>Total Messages</p></div>", unsafe_allow_html=True)

with m3:
    cam_status = "ON" if status["active"] else "OFF"
    st.markdown(f"<div class='metric-card'><h3>{cam_status}</h3><p>Camera Status</p></div>", unsafe_allow_html=True)

with m4:
    st.markdown("<div class='metric-card'><h3>LIVE</h3><p>AI Monitoring</p></div>", unsafe_allow_html=True)

st.write("")

# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns([2.1, 1])

# =========================================================
# CHAT SECTION
# =========================================================

with left:
    st.markdown("## 💬 Live Multi-Client Chat")

    chat_container = st.container(height=560)

    with chat_container:
        messages = load_messages()

        if len(messages) == 0:
            st.info("No messages yet")
        else:
            for msg in reversed(messages):
                st.markdown(
                    f"""
                    <div class="chat-card">
                        <div class="chat-user">👤 {msg['user']}</div>
                        <div class="chat-text">💬 {msg['text']}</div>
                        <span class="result-pill">Emotion : {msg['emotion']}</span>
                        <span class="result-pill">Sarcasm : {msg['sarcasm']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    text = st.chat_input("Type your message...")

    if text:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text},
                timeout=20
            )

            data = response.json()

            save_message({
                "user": username,
                "text": text,
                "emotion": data["emotion"],
                "sarcasm": data["sarcasm"]
            })

            st.rerun()

        except Exception as e:
            st.error(f"FastAPI Error : {e}")

# =========================================================
# VIDEO SECTION
# =========================================================

with right:
    st.markdown("## 🎥 Live Facial Emotion")
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if not status["active"]:
        if st.button("🎥 Start Camera", use_container_width=True):
            granted, message = request_camera(username)

            if granted:
                st.session_state.video_active = True
                st.session_state.camera_user = username
                st.rerun()
            else:
                st.error(message)

    elif status["user"] != username:
        st.warning(f"🎥 Live Camera : {status['user']}")

        if os.path.exists("shared_camera.jpg"):
            image = cv2.imread("shared_camera.jpg")

            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, channels="RGB", use_container_width=True)

        if os.path.exists("shared_emotion.json"):
            try:
                with open("shared_emotion.json", "r", encoding="utf-8") as f:
                    emotions = json.load(f)

                st.markdown("### 🎭 Live Emotions")

                for emo in emotions[:10]:
                    st.markdown(
                        f"""
                        <div class="emotion-box">
                            {emo['emotion']} ({emo['confidence']:.1f}%)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except:
                pass

    else:
        st.success("Camera Running")

        if st.button("❌ Stop Camera", use_container_width=True):
            release_camera(username)
            st.session_state.video_active = False
            st.session_state.camera_user = None
            st.rerun()

        cap, msg = start_camera()

        if cap is None:
            st.error(msg)
        else:
            FRAME_WINDOW = st.empty()
            emotion_placeholder = st.empty()

            while True:
                current_status = get_camera_status()

                if not current_status["active"] or current_status["user"] != username:
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                try:
                    frame, emotions = detect_emotion(frame)
                except Exception as e:
                    st.error(f"Emotion Error : {e}")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                FRAME_WINDOW.image(
                    frame,
                    channels="RGB",
                    use_container_width=True
                )

                if len(emotions) > 0:
                    emo = emotions[0]["emotion"]
                    conf = emotions[0]["confidence"]
                    latest = f"{emo} ({conf:.1f}%)"

                    if len(st.session_state.video_emotions) == 0 or st.session_state.video_emotions[-1] != latest:
                        st.session_state.video_emotions.append(latest)

                with emotion_placeholder.container():
                    st.markdown("### 🎭 Recent Facial Emotions")

                    for item in reversed(st.session_state.video_emotions):
                        st.markdown(
                            f"""
                            <div class="emotion-box">
                                {item}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                time.sleep(0.03)

            stop_camera(cap)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ANALYTICS
# =========================================================

st.divider()
st.markdown("## 📊 Analytics Dashboard")

messages = load_messages()

if len(messages) > 0:
    df = pd.DataFrame(messages)

    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Emotion Distribution")
        emotion_counts = df["emotion"].value_counts()
        st.bar_chart(emotion_counts)

    with c2:
        st.markdown("### Sarcasm Distribution")
        sarcasm_counts = df["sarcasm"].value_counts()
        st.bar_chart(sarcasm_counts)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📂 Export Chat With Emotion",
        data=csv,
        file_name="emotion_chat_export.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("No analytics yet")





































# # =========================================================
# # FINAL INDUSTRY STYLE app.py
# # MULTI CLIENT + SHARED VIDEO + EMOTION SYSTEM
# # =========================================================

# import json
# import os
# import time
# import cv2
# import requests
# import pandas as pd
# import streamlit as st
# import streamlit.components.v1 as components

# from collections import deque

# from webcam import start_camera, stop_camera, detect_emotion

# from camera_manager import (
#     request_camera,
#     release_camera,
#     get_camera_status
# )

# # =========================================================
# # PAGE CONFIG
# # =========================================================

# st.set_page_config(
#     page_title="Distributed AI Emotion System",
#     page_icon="🤖",
#     layout="wide"
# )

# # =========================================================
# # CSS
# # =========================================================

# st.markdown("""
# <style>

# .stApp {
#     background:
#         radial-gradient(circle at top left, #12203a 0%, #080d18 35%, #05070d 100%);
#     color: white;
# }

# .block-container {
#     padding-top: 2rem !important;
#     padding-left: 3rem !important;
#     padding-right: 3rem !important;
# }

# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #0f172a, #020617);
#     border-right: 1px solid #1e293b;
# }

# .hero-box {
#     padding: 28px;
#     border-radius: 26px;
#     background: linear-gradient(135deg, rgba(0,255,213,0.16), rgba(37,99,235,0.15));
#     border: 1px solid rgba(0,255,213,0.35);
#     box-shadow: 0 0 35px rgba(0,255,213,0.12);
#     margin-bottom: 25px;
# }

# .main-title {
#     font-size: 42px;
#     font-weight: 900;
#     letter-spacing: 1px;
#     background: linear-gradient(90deg, #00ffd5, #38bdf8, #a78bfa);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     line-height: 1.15;
# }

# .sub-title {
#     color: #cbd5e1;
#     font-size: 17px;
#     margin-top: 8px;
# }

# .glass-card {
#     background: rgba(15, 23, 42, 0.82);
#     border: 1px solid rgba(148, 163, 184, 0.22);
#     border-radius: 22px;
#     padding: 18px;
#     box-shadow: 0 12px 30px rgba(0,0,0,0.35);
# }

# .chat-card {
#     background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
#     padding: 18px;
#     border-radius: 18px;
#     margin-bottom: 15px;
#     border-left: 5px solid #00ffd5;
#     box-shadow: 0 8px 20px rgba(0,0,0,0.28);
# }

# .chat-user {
#     font-size: 15px;
#     color: #7dd3fc;
#     font-weight: 700;
# }

# .chat-text {
#     font-size: 16px;
#     color: #f8fafc;
#     margin-top: 12px;
#     line-height: 1.6;
# }

# .result-pill {
#     display: inline-block;
#     padding: 8px 13px;
#     margin-top: 10px;
#     margin-right: 8px;
#     border-radius: 999px;
#     background: rgba(0,255,213,0.12);
#     border: 1px solid rgba(0,255,213,0.35);
#     color: #e0f2fe;
#     font-weight: 600;
# }

# .emotion-box {
#     background: rgba(2, 6, 23, 0.85);
#     padding: 13px;
#     border-radius: 14px;
#     margin-bottom: 10px;
#     border: 1px solid #334155;
# }

# .status-card {
#     padding: 14px;
#     border-radius: 16px;
#     background: rgba(34,197,94,0.13);
#     border: 1px solid rgba(34,197,94,0.3);
#     margin-bottom: 12px;
# }

# .warning-card {
#     padding: 14px;
#     border-radius: 16px;
#     background: rgba(239,68,68,0.13);
#     border: 1px solid rgba(239,68,68,0.3);
#     margin-bottom: 12px;
# }

# .metric-card {
#     background: rgba(15,23,42,0.9);
#     border: 1px solid rgba(148,163,184,0.25);
#     border-radius: 20px;
#     padding: 20px;
#     text-align: center;
# }

# .robot-note {
#     color: #94a3b8;
#     font-size: 13px;
# }

# div[data-testid="stDataFrame"] {
#     border-radius: 18px;
#     overflow: hidden;
# }

# </style>
# """, unsafe_allow_html=True)

# # =========================================================
# # ROBOT MASCOT COMPONENT
# # =========================================================

# components.html("""
# <div id="botBox">
#     <div id="robot">
#         <div class="antenna"></div>
#         <div class="head">
#             <div class="eye left"></div>
#             <div class="eye right"></div>
#             <div class="mouth"></div>
#         </div>
#         <div class="body">AI</div>
#         <div class="hand">👉</div>
#     </div>
#     <div class="msg">I am watching your cursor 👀</div>
# </div>

# <style>
# #botBox {
#     height: 135px;
#     width: 100%;
#     position: relative;
#     overflow: hidden;
#     background: linear-gradient(90deg, #020617, #0f172a);
#     border: 1px solid rgba(0,255,213,0.3);
#     border-radius: 22px;
#     box-shadow: 0 0 25px rgba(0,255,213,0.12);
# }

# #robot {
#     position: absolute;
#     left: 30px;
#     top: 18px;
#     transition: all 0.15s ease;
# }

# .antenna {
#     width: 4px;
#     height: 18px;
#     background: #00ffd5;
#     margin-left: 42px;
#     border-radius: 8px;
# }

# .head {
#     width: 90px;
#     height: 58px;
#     background: linear-gradient(135deg, #38bdf8, #8b5cf6);
#     border-radius: 20px;
#     position: relative;
#     box-shadow: 0 0 20px rgba(56,189,248,0.5);
# }

# .eye {
#     width: 13px;
#     height: 13px;
#     background: #020617;
#     border-radius: 50%;
#     position: absolute;
#     top: 18px;
# }

# .left { left: 22px; }
# .right { right: 22px; }

# .mouth {
#     width: 32px;
#     height: 6px;
#     background: #020617;
#     border-radius: 10px;
#     position: absolute;
#     left: 29px;
#     bottom: 13px;
# }

# .body {
#     width: 70px;
#     height: 35px;
#     background: #00ffd5;
#     color: #020617;
#     font-weight: 900;
#     text-align: center;
#     line-height: 35px;
#     margin-left: 10px;
#     border-radius: 14px;
#     margin-top: 5px;
# }

# .hand {
#     position: absolute;
#     left: 88px;
#     top: 55px;
#     font-size: 30px;
#     transform-origin: left center;
#     animation: wave 1s infinite alternate;
# }

# .msg {
#     position: absolute;
#     left: 160px;
#     top: 48px;
#     font-family: Segoe UI;
#     font-weight: 700;
#     color: #e0f2fe;
#     font-size: 20px;
# }

# @keyframes wave {
#     from { transform: rotate(-10deg); }
#     to { transform: rotate(20deg); }
# }
# </style>

# <script>
# const box = document.getElementById("botBox");
# const robot = document.getElementById("robot");

# box.addEventListener("mousemove", (e) => {
#     const rect = box.getBoundingClientRect();
#     let x = e.clientX - rect.left - 55;
#     if (x < 20) x = 20;
#     if (x > rect.width - 140) x = rect.width - 140;
#     robot.style.left = x + "px";
# });
# </script>
# """, height=150)

# # =========================================================
# # FILES
# # =========================================================

# CHAT_FILE = "global_chat.json"
# USERS_FILE = "online_users.json"

# if not os.path.exists(CHAT_FILE):
#     with open(CHAT_FILE, "w", encoding="utf-8") as f:
#         json.dump([], f)

# if not os.path.exists(USERS_FILE):
#     with open(USERS_FILE, "w", encoding="utf-8") as f:
#         json.dump([], f)

# # =========================================================
# # FUNCTIONS
# # =========================================================

# def load_messages():
#     try:
#         with open(CHAT_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except:
#         return []

# def save_message(msg):
#     messages = load_messages()
#     messages.append(msg)
#     messages = messages[-100:]
#     with open(CHAT_FILE, "w", encoding="utf-8") as f:
#         json.dump(messages, f, indent=4, ensure_ascii=False)

# def load_users():
#     try:
#         with open(USERS_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except:
#         return []

# def save_users(users):
#     with open(USERS_FILE, "w", encoding="utf-8") as f:
#         json.dump(users, f, indent=4, ensure_ascii=False)

# # =========================================================
# # SESSION STATE
# # =========================================================

# if "video_active" not in st.session_state:
#     st.session_state.video_active = False

# if "camera_user" not in st.session_state:
#     st.session_state.camera_user = None

# if "video_emotions" not in st.session_state:
#     st.session_state.video_emotions = deque(maxlen=10)

# # =========================================================
# # HEADER
# # =========================================================

# st.markdown("""
# <div class="hero-box">
#     <div class="main-title">🔥 Distributed AI Emotion Detection System</div>
#     <div class="sub-title">
#         Live Multi-Client Chat • Facial Emotion Recognition • Sarcasm Detection • Real-Time Analytics
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # =========================================================
# # SIDEBAR
# # =========================================================

# st.sidebar.markdown("## 🤖 User Control Panel")

# username = st.sidebar.text_input("Enter Username", value="Sankalp")

# st.sidebar.markdown(
#     f"<div class='status-card'>✅ Connected : <b>{username}</b></div>",
#     unsafe_allow_html=True
# )

# users = load_users()
# if username not in users:
#     users.append(username)
#     save_users(users)

# status = get_camera_status()

# if status["active"]:
#     st.sidebar.markdown(
#         f"<div class='warning-card'>🎥 Camera Active : <b>{status['user']}</b></div>",
#         unsafe_allow_html=True
#     )
# else:
#     st.sidebar.markdown(
#         "<div class='status-card'>🎥 Camera Available</div>",
#         unsafe_allow_html=True
#     )

# st.sidebar.markdown("## 🟢 Online Users")

# for user in users:
#     if status["active"] and status["user"] == user:
#         st.sidebar.markdown(f"🎥 **{user}**  <span style='color:#22c55e'>(VIDEO ON)</span>", unsafe_allow_html=True)
#     else:
#         st.sidebar.markdown(f"📴 **{user}**  <span style='color:#ef4444'>(VIDEO OFF)</span>", unsafe_allow_html=True)

# # =========================================================
# # TOP METRICS
# # =========================================================

# messages = load_messages()

# m1, m2, m3, m4 = st.columns(4)

# with m1:
#     st.markdown(f"<div class='metric-card'><h3>{len(users)}</h3><p>Online Users</p></div>", unsafe_allow_html=True)

# with m2:
#     st.markdown(f"<div class='metric-card'><h3>{len(messages)}</h3><p>Total Messages</p></div>", unsafe_allow_html=True)

# with m3:
#     cam_status = "ON" if status["active"] else "OFF"
#     st.markdown(f"<div class='metric-card'><h3>{cam_status}</h3><p>Camera Status</p></div>", unsafe_allow_html=True)

# with m4:
#     st.markdown("<div class='metric-card'><h3>LIVE</h3><p>AI Monitoring</p></div>", unsafe_allow_html=True)

# st.write("")

# # =========================================================
# # MAIN LAYOUT
# # =========================================================

# left, right = st.columns([2.1, 1])

# # =========================================================
# # CHAT SECTION
# # =========================================================

# with left:
#     st.markdown("## 💬 Live Multi-Client Chat")

#     chat_container = st.container(height=560)

#     with chat_container:
#         messages = load_messages()

#         if len(messages) == 0:
#             st.info("No messages yet")
#         else:
#             for msg in reversed(messages):
#                 st.markdown(
#                     f"""
#                     <div class="chat-card">
#                         <div class="chat-user">👤 {msg['user']}</div>
#                         <div class="chat-text">💬 {msg['text']}</div>
#                         <span class="result-pill">Emotion : {msg['emotion']}</span>
#                         <span class="result-pill">Sarcasm : {msg['sarcasm']}</span>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

#     text = st.chat_input("Type your message...")

#     if text:
#         try:
#             response = requests.post(
#                 "http://127.0.0.1:8000/predict",
#                 json={"text": text},
#                 timeout=20
#             )

#             data = response.json()

#             save_message({
#                 "user": username,
#                 "text": text,
#                 "emotion": data["emotion"],
#                 "sarcasm": data["sarcasm"]
#             })

#             st.rerun()

#         except Exception as e:
#             st.error(f"FastAPI Error : {e}")

# # =========================================================
# # VIDEO SECTION
# # =========================================================

# with right:
#     st.markdown("## 🎥 Live Facial Emotion")
#     st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

#     if not status["active"]:
#         if st.button("🎥 Start Camera", use_container_width=True):
#             granted, message = request_camera(username)

#             if granted:
#                 st.session_state.video_active = True
#                 st.session_state.camera_user = username
#                 st.rerun()
#             else:
#                 st.error(message)

#     elif status["user"] != username:
#         st.warning(f"🎥 Live Camera : {status['user']}")

#         if os.path.exists("shared_camera.jpg"):
#             image = cv2.imread("shared_camera.jpg")

#             if image is not None:
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 st.image(image, channels="RGB", use_container_width=True)

#         if os.path.exists("shared_emotion.json"):
#             try:
#                 with open("shared_emotion.json", "r", encoding="utf-8") as f:
#                     emotions = json.load(f)

#                 st.markdown("### 🎭 Live Emotions")

#                 for emo in emotions[:10]:
#                     st.markdown(
#                         f"""
#                         <div class="emotion-box">
#                             {emo['emotion']} ({emo['confidence']:.1f}%)
#                         </div>
#                         """,
#                         unsafe_allow_html=True
#                     )

#             except:
#                 pass

#     else:
#         st.success("Camera Running")

#         if st.button("❌ Stop Camera", use_container_width=True):
#             release_camera(username)
#             st.session_state.video_active = False
#             st.session_state.camera_user = None
#             st.rerun()

#         cap, msg = start_camera()

#         if cap is None:
#             st.error(msg)
#         else:
#             FRAME_WINDOW = st.empty()
#             emotion_placeholder = st.empty()

#             while True:
#                 current_status = get_camera_status()

#                 if not current_status["active"] or current_status["user"] != username:
#                     break

#                 ret, frame = cap.read()

#                 if not ret:
#                     break

#                 try:
#                     frame, emotions = detect_emotion(frame)
#                 except Exception as e:
#                     st.error(f"Emotion Error : {e}")
#                     break

#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                 FRAME_WINDOW.image(
#                     frame,
#                     channels="RGB",
#                     use_container_width=True
#                 )

#                 if len(emotions) > 0:
#                     emo = emotions[0]["emotion"]
#                     conf = emotions[0]["confidence"]
#                     latest = f"{emo} ({conf:.1f}%)"

#                     if len(st.session_state.video_emotions) == 0 or st.session_state.video_emotions[-1] != latest:
#                         st.session_state.video_emotions.append(latest)

#                 with emotion_placeholder.container():
#                     st.markdown("### 🎭 Recent Facial Emotions")

#                     for item in reversed(st.session_state.video_emotions):
#                         st.markdown(
#                             f"""
#                             <div class="emotion-box">
#                                 {item}
#                             </div>
#                             """,
#                             unsafe_allow_html=True
#                         )

#                 time.sleep(0.03)

#             stop_camera(cap)

#     st.markdown("</div>", unsafe_allow_html=True)

# # =========================================================
# # ANALYTICS
# # =========================================================

# st.divider()
# st.markdown("## 📊 Analytics Dashboard")

# messages = load_messages()

# if len(messages) > 0:
#     df = pd.DataFrame(messages)

#     st.dataframe(df, use_container_width=True)

#     c1, c2 = st.columns(2)

#     with c1:
#         st.markdown("### Emotion Distribution")
#         emotion_counts = df["emotion"].value_counts()
#         st.bar_chart(emotion_counts)

#     with c2:
#         st.markdown("### Sarcasm Distribution")
#         sarcasm_counts = df["sarcasm"].value_counts()
#         st.bar_chart(sarcasm_counts)

#     csv = df.to_csv(index=False).encode("utf-8")

#     st.download_button(
#         label="📂 Export Chat With Emotion",
#         data=csv,
#         file_name="emotion_chat_export.csv",
#         mime="text/csv",
#         use_container_width=True
#     )

# else:
#     st.info("No analytics yet")
















































#  # =========================================================
# # FINAL COMPLETE app.py
# # MULTI CLIENT + SHARED VIDEO + EMOTION SYSTEM
# # =========================================================

# import json
# import os
# import time
# import cv2
# import requests
# import pandas as pd
# import streamlit as st

# from collections import deque

# from webcam import (
#     start_camera,
#     stop_camera,
#     detect_emotion
# )

# from camera_manager import (
#     request_camera,
#     release_camera,
#     get_camera_status
# )

# # =========================================================
# # PAGE CONFIG
# # =========================================================

# st.set_page_config(
#     page_title="Distributed AI Emotion System",
#     page_icon="🤖",
#     layout="wide"
# )

# # =========================================================
# # CSS
# # =========================================================

# st.markdown("""
# <style>

# html, body, [class*="css"]{
#     background-color:#0b1120;
#     color:white;
#     font-family:'Segoe UI';
# }

# .block-container{
#     padding-top:1rem;
# }

# .main-title{
#     font-size:42px;
#     font-weight:800;
#     color:#00ffd5;
# }

# .sub-title{
#     color:#8b949e;
#     margin-bottom:20px;
# }

# .chat-card{
#     background:#172033;
#     padding:14px;
#     border-radius:15px;
#     margin-bottom:12px;
#     border-left:4px solid #00ffd5;
# }

# .camera-active{
#     color:#00ff99;
#     font-weight:bold;
# }

# .camera-off{
#     color:#ff4b4b;
#     font-weight:bold;
# }

# .emotion-box{
#     background:#111827;
#     padding:10px;
#     border-radius:12px;
#     margin-bottom:10px;
#     border:1px solid #1f2937;
# }

# </style>
# """, unsafe_allow_html=True)

# # =========================================================
# # FILES
# # =========================================================

# CHAT_FILE = "global_chat.json"
# USERS_FILE = "online_users.json"

# # =========================================================
# # CREATE FILES
# # =========================================================

# if not os.path.exists(CHAT_FILE):
#     with open(CHAT_FILE, "w") as f:
#         json.dump([], f)

# if not os.path.exists(USERS_FILE):
#     with open(USERS_FILE, "w") as f:
#         json.dump([], f)

# # =========================================================
# # CHAT FUNCTIONS
# # =========================================================

# def load_messages():

#     try:
#         with open(CHAT_FILE, "r") as f:
#             return json.load(f)

#     except:
#         return []

# # =========================================================

# def save_message(msg):

#     messages = load_messages()

#     messages.append(msg)

#     # LAST 100 ONLY
#     messages = messages[-100:]

#     with open(CHAT_FILE, "w") as f:
#         json.dump(messages, f, indent=4)

# # =========================================================
# # USER FUNCTIONS
# # =========================================================

# def load_users():

#     try:
#         with open(USERS_FILE, "r") as f:
#             return json.load(f)

#     except:
#         return []

# # =========================================================

# def save_users(users):

#     with open(USERS_FILE, "w") as f:
#         json.dump(users, f, indent=4)

# # =========================================================
# # SESSION STATE
# # =========================================================

# if "video_active" not in st.session_state:
#     st.session_state.video_active = False

# if "camera_user" not in st.session_state:
#     st.session_state.camera_user = None

# if "video_emotions" not in st.session_state:
#     st.session_state.video_emotions = deque(maxlen=10)

# # =========================================================
# # HEADER
# # =========================================================

# st.markdown(
#     """
#     <div class="main-title">
#         🔥 Distributed AI Emotion Detection System
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown(
#     """
#     <div class="sub-title">
#         Live Multi-Client Chat + Facial Emotion + Sarcasm Detection
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# st.divider()

# # =========================================================
# # USERNAME
# # =========================================================

# username = st.sidebar.text_input(
#     "Enter Username",
#     value="Sankalp"
# )

# st.sidebar.success(f"Connected : {username}")

# # =========================================================
# # SAVE USER
# # =========================================================

# users = load_users()

# if username not in users:

#     users.append(username)
#     save_users(users)

# # =========================================================
# # CAMERA STATUS
# # =========================================================

# status = get_camera_status()

# if status["active"]:

#     st.sidebar.warning(
#         f"🎥 Camera Active : {status['user']}"
#     )

# else:

#     st.sidebar.success(
#         "🎥 Camera Available"
#     )

# # =========================================================
# # ONLINE USERS
# # =========================================================

# st.sidebar.markdown("## 🟢 Online Users")

# for user in users:

#     if status["active"] and status["user"] == user:

#         st.sidebar.markdown(
#             f"""
#             🎥 {user}
#             <span class='camera-active'>
#                 (VIDEO ON)
#             </span>
#             """,
#             unsafe_allow_html=True
#         )

#     else:

#         st.sidebar.markdown(
#             f"""
#             📴 {user}
#             <span class='camera-off'>
#                 (VIDEO OFF)
#             </span>
#             """,
#             unsafe_allow_html=True
#         )

# # =========================================================
# # MAIN LAYOUT
# # =========================================================

# left, right = st.columns([2, 1])

# # =========================================================
# # LEFT SIDE CHAT
# # =========================================================

# with left:

#     st.subheader("💬 Live Multi-Client Chat")

#     chat_container = st.container(height=550)

#     with chat_container:

#         messages = load_messages()

#         if len(messages) == 0:

#             st.info("No messages yet")

#         else:

#             for msg in reversed(messages):

#                 st.markdown(
#                     f"""
#                     <div class="chat-card">

#                         <b>👤 {msg['user']}</b>

#                         <br><br>

#                         💬 {msg['text']}

#                         <br><br>

#                           Emotion : {msg['emotion']}

#                         <br>

#                           Sarcasm : {msg['sarcasm']}

#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

#     # =====================================================
#     # CHAT INPUT
#     # =====================================================

#     text = st.chat_input(
#         "Type your message..."
#     )

#     if text:

#         try:

#             response = requests.post(
#                 "http://127.0.0.1:8000/predict",
#                 json={
#                     "text": text
#                 }
#             )

#             data = response.json()

#             save_message({

#                 "user": username,
#                 "text": text,
#                 "emotion": data["emotion"],
#                 "sarcasm": data["sarcasm"]

#             })

#             st.rerun()

#         except Exception as e:

#             st.error(
#                 f"FastAPI Error : {e}"
#             )

# # =========================================================
# # RIGHT SIDE VIDEO
# # =========================================================

# with right:

#     st.subheader("🎥 Live Facial Emotion")

#     # =====================================================
#     # CAMERA AVAILABLE
#     # =====================================================

#     if not status["active"]:

#         if st.button("🎥 Start Camera"):

#             granted, message = request_camera(username)

#             if granted:

#                 st.session_state.video_active = True
#                 st.session_state.camera_user = username

#                 st.rerun()

#             else:

#                 st.error(message)

#     # =====================================================
#     # SOMEONE ELSE USING CAMERA
#     # =====================================================

#     elif status["user"] != username:

#         st.warning(
#             f"🎥 Live Camera : {status['user']}"
#         )

#         # =================================================
#         # SHOW SHARED VIDEO
#         # =================================================

#         if os.path.exists("shared_camera.jpg"):

#             image = cv2.imread(
#                 "shared_camera.jpg"
#             )

#             if image is not None:

#                 image = cv2.cvtColor(
#                     image,
#                     cv2.COLOR_BGR2RGB
#                 )

#                 st.image(
#                     image,
#                     channels="RGB"
#                 )

#         # =================================================
#         # SHOW SHARED EMOTIONS
#         # =================================================

#         if os.path.exists("shared_emotion.json"):

#             try:

#                 with open(
#                     "shared_emotion.json",
#                     "r"
#                 ) as f:

#                     emotions = json.load(f)

#                 st.markdown(
#                     "### 🎭 Live Emotions"
#                 )

#                 for emo in emotions[:10]:

#                     st.markdown(
#                         f"""
#                         <div class="emotion-box">

#                             {emo['emotion']}
#                             ({emo['confidence']:.1f}%)

#                         </div>
#                         """,
#                         unsafe_allow_html=True
#                     )

#             except:
#                 pass

#     # =====================================================
#     # CURRENT USER CAMERA
#     # =====================================================

#     else:

#         st.success("Camera Running")

#         if st.button("❌ Stop Camera"):

#             release_camera(username)

#             st.session_state.video_active = False
#             st.session_state.camera_user = None

#             st.rerun()

#         # =================================================
#         # START CAMERA
#         # =================================================

#         cap, msg = start_camera()

#         if cap is None:

#             st.error(msg)

#         else:

#             FRAME_WINDOW = st.empty()

#             emotion_placeholder = st.empty()

#             while True:

#                 current_status = get_camera_status()

#                 if (
#                     not current_status["active"]
#                     or current_status["user"] != username
#                 ):
#                     break

#                 ret, frame = cap.read()

#                 if not ret:
#                     break

#                 # =========================================
#                 # DETECT EMOTION
#                 # =========================================

#                 try:

#                     frame, emotions = detect_emotion(frame)

#                 except Exception as e:

#                     st.error(
#                         f"Emotion Error : {e}"
#                     )

#                     break

#                 # =========================================
#                 # RGB
#                 # =========================================

#                 frame = cv2.cvtColor(
#                     frame,
#                     cv2.COLOR_BGR2RGB
#                 )

#                 FRAME_WINDOW.image(
#                     frame,
#                     channels="RGB"
#                 )

#                 # =========================================
#                 # LAST 10 EMOTIONS
#                 # =========================================

#                 if len(emotions) > 0:

#                     emo = emotions[0]["emotion"]
#                     conf = emotions[0]["confidence"]

#                     latest = (
#                         f"{emo} ({conf:.1f}%)"
#                     )

#                     if (
#                         len(st.session_state.video_emotions) == 0
#                         or
#                         st.session_state.video_emotions[-1] != latest
#                     ):

#                         st.session_state.video_emotions.append(
#                             latest
#                         )

#                 # =========================================
#                 # SHOW LAST 10
#                 # =========================================

#                 with emotion_placeholder.container():

#                     st.markdown(
#                         "### 🎭 Recent Facial Emotions"
#                     )

#                     for item in reversed(
#                         st.session_state.video_emotions
#                     ):

#                         st.markdown(
#                             f"""
#                             <div class="emotion-box">
#                                 {item}
#                             </div>
#                             """,
#                             unsafe_allow_html=True
#                         )

#                 time.sleep(0.03)

#             stop_camera(cap)

# # =========================================================
# # ANALYTICS
# # =========================================================

# st.divider()

# st.subheader("📊 Analytics Dashboard")

# messages = load_messages()

# if len(messages) > 0:

#     df = pd.DataFrame(messages)

#     st.dataframe(
#         df,
#         use_container_width=True
#     )

#     st.subheader(
#         "Emotion Distribution"
#     )

#     emotion_counts = (
#         df["emotion"]
#         .value_counts()
#     )

#     st.bar_chart(
#         emotion_counts
#     )

#     # =====================================================
#     # EXPORT CHAT
#     # =====================================================

#     csv = df.to_csv(
#         index=False
#     ).encode("utf-8")

#     st.download_button(
#         label="📂 Export Chat With Emotion",
#         data=csv,
#         file_name="emotion_chat_export.csv",
#         mime="text/csv"
#     )

# else:

#     st.info(
#         "No analytics yet"
#     )
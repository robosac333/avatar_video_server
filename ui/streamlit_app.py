"""
Streamlit front-end for the avatar lip-sync server.

Run it pointing at wherever the FastAPI server lives:
    SERVER_URL=http://localhost:8000 streamlit run ui/streamlit_app.py

On RunPod the server and UI run on the same pod, so localhost is correct; you
just expose the Streamlit port (8501) via the pod's HTTP proxy.
"""
import os
import time
import requests
import streamlit as st

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Avatar Studio", page_icon="🎬", layout="centered")

# ── Minimal, calm styling ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .stApp { background: #0f1115; }
      h1, h2, h3, p, label, .stMarkdown { color: #e8e6e3 !important; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:12px; font-weight:600; }
      .ok   { background:#10371f; color:#6ee7a0; }
      .run  { background:#3a2f10; color:#f5c451; }
      .err  { background:#3a1414; color:#f08a8a; }
      .muted{ color:#8a8f98 !important; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Avatar Studio")
st.markdown('<p class="muted">Self-hosted lip-sync — image + audio + prompt → talking reel.</p>',
            unsafe_allow_html=True)

# ── Server health ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Server")
    st.code(SERVER_URL, language=None)
    try:
        h = requests.get(f"{SERVER_URL}/healthz", timeout=5).json()
        installed = h.get("models_installed", {})
        st.markdown(f"Queue depth: **{h.get('queue_depth', '?')}**")
        for m, ok in installed.items():
            mark = "✅" if ok else "⛔"
            st.markdown(f"{mark} `{m}` weights")
    except Exception:
        st.error("Server unreachable. Is the FastAPI process running?")

# ── Input form ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("Portrait image", type=["png", "jpg", "jpeg", "webp"])
with col2:
    audio_file = st.file_uploader("Voiceover audio", type=["mp3", "wav", "m4a"])

if image_file:
    st.image(image_file, caption="Avatar", use_container_width=True)
if audio_file:
    st.audio(audio_file)

model = st.selectbox(
    "Model",
    options=["multitalk", "hunyuan"],
    format_func=lambda m: {
        "multitalk": "Wan 2.1 MultiTalk — best for everyday reels (faster, lighter)",
        "hunyuan":   "HunyuanVideo-Avatar — higher fidelity hero content (heavier)",
    }[m],
)
resolution = st.radio("Resolution", ["480", "720"], horizontal=True,
                      help="480 renders roughly twice as fast. Use it for drafts.")

prompt = st.text_area(
    "Motion / expression prompt",
    value=("Smiling confidently, she speaks into the microphone, looking directly "
           "at the camera. Natural head and hand movements, occasional slight nod, "
           "warm smile maintained throughout with professional, engaged presenter "
           "energy. At moments of emphasis, brighten facial expression — wider "
           "smile, raised eyebrows, eyes brightening with enthusiasm."),
    height=140,
)

go = st.button("Generate video", type="primary", use_container_width=True,
               disabled=not (image_file and audio_file))

# ── Submit + poll ───────────────────────────────────────────────────────────
if go:
    files = {
        "image": (image_file.name, image_file.getvalue()),
        "audio": (audio_file.name, audio_file.getvalue()),
    }
    data = {"prompt": prompt, "model": model, "resolution": resolution}
    try:
        r = requests.post(f"{SERVER_URL}/generate", files=files, data=data, timeout=60)
        r.raise_for_status()
        job = r.json()
    except Exception as exc:
        st.error(f"Could not submit job: {exc}")
        st.stop()

    job_id = job["job_id"]
    st.markdown(f"Job **`{job_id}`** queued — audio {job.get('audio_seconds','?')}s.")
    status_box = st.empty()
    progress = st.progress(0)

    t0 = time.time()
    while True:
        try:
            s = requests.get(f"{SERVER_URL}/status/{job_id}", timeout=10).json()
        except Exception as exc:
            status_box.error(f"Lost contact with server: {exc}")
            break

        state = s.get("status")
        elapsed = int(time.time() - t0)
        if state == "queued":
            status_box.markdown('<span class="pill run">queued</span>'
                                 f' <span class="muted">{elapsed}s</span>',
                                 unsafe_allow_html=True)
            progress.progress(10)
        elif state == "running":
            status_box.markdown('<span class="pill run">rendering</span>'
                                 f' <span class="muted">{elapsed}s elapsed</span>',
                                 unsafe_allow_html=True)
            # crude visual creep so the bar feels alive during the long render
            progress.progress(min(90, 10 + elapsed))
        elif state == "done":
            progress.progress(100)
            status_box.markdown('<span class="pill ok">done</span>'
                                 f' <span class="muted">in {s.get("duration_s","?")}s</span>',
                                 unsafe_allow_html=True)
            vid = requests.get(f"{SERVER_URL}/result/{job_id}", timeout=60).content
            st.video(vid)
            st.download_button("Download mp4", vid, file_name=f"{job_id}.mp4",
                               mime="video/mp4", use_container_width=True)
            break
        elif state == "failed":
            status_box.markdown('<span class="pill err">failed</span>',
                                 unsafe_allow_html=True)
            st.error(s.get("error", "unknown error"))
            break
        time.sleep(3)

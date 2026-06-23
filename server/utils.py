"""Small helpers: audio duration probing, safe filenames, image prep."""
import subprocess
import uuid
import json
from pathlib import Path
from PIL import Image


def audio_duration_seconds(audio_path: Path) -> float:
    """Return audio length in seconds using ffprobe (ships with ffmpeg)."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def prep_image(src: Path, dst: Path, target_short_side: int = 480) -> Path:
    """
    Normalise the input portrait: convert to RGB, downscale so the short side
    matches the target resolution. The avatar models choke on huge images and
    on alpha channels, so we sanitise before handing it over.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    short = min(w, h)
    if short > target_short_side:
        scale = target_short_side / short
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=95)
    return dst


def run_logged(cmd: list[str], cwd: Path, log_path: Path, extra_env: dict | None = None) -> int:
    """
    Run a subprocess, streaming combined stdout/stderr to a log file so a job's
    progress is inspectable while it runs and afterwards for debugging.
    Returns the process exit code.
    """
    import os
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
        proc.wait()
    return proc.returncode

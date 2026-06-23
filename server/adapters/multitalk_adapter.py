"""
Adapter for MeiGen-AI/MultiTalk (Wan 2.1 based).

The official repo's supported entry point is `generate_multitalk.py`, driven by
an input JSON that points at the image + audio and carries the prompt. We build
that JSON, invoke the script with the VRAM/speed flags from config, and hand
back the produced mp4. Wrapping the script (rather than importing internals)
means upstream refactors don't break us.

Reference: https://github.com/MeiGen-AI/MultiTalk
"""
import json
import sys
from pathlib import Path

from .. import config
from ..utils import run_logged


def build_input_json(job_dir: Path, image_path: Path, audio_path: Path, prompt: str) -> Path:
    """
    MultiTalk expects a JSON describing the speaker(s). Single-speaker reel =
    one entry under `cond_audio`. The `prompt` steers scene/expression while the
    audio drives the lips.
    """
    cfg = {
        "prompt": prompt,
        "cond_image": str(image_path),
        "cond_audio": {"person1": str(audio_path)},
    }
    out = job_dir / "input.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def generate(job_dir: Path, image_path: Path, audio_path: Path, prompt: str,
             resolution: str = "480") -> Path:
    """
    Run MultiTalk. Returns the path to the generated mp4.
    Raises RuntimeError on non-zero exit (caller marks the job failed).
    """
    input_json = build_input_json(job_dir, image_path, audio_path, prompt)
    out_stem = job_dir / "result"          # script appends .mp4
    log_path = job_dir / "generate.log"

    cmd = [
        sys.executable, "generate_multitalk.py",
        "--ckpt_dir", str(config.WAN_BASE_CKPT),
        "--wav2vec_dir", str(config.WAV2VEC_DIR),
        "--input_json", str(input_json),
        "--sample_steps", str(config.SAMPLE_STEPS),
        "--mode", "streaming",             # lets longer audio generate in chunks
        "--size", f"multitalk-{resolution}",
        "--num_persistent_param_in_dit", config.NUM_PERSISTENT_PARAM_IN_DIT,
        "--save_file", str(out_stem),
    ]
    if config.ENABLE_TEACACHE:
        cmd += ["--use_teacache", "--teacache_thresh", config.TEACACHE_THRESHOLD]

    code = run_logged(
        cmd, cwd=config.MULTITALK_REPO, log_path=log_path,
        extra_env={"CUDA_VISIBLE_DEVICES": config.CUDA_VISIBLE_DEVICES},
    )
    if code != 0:
        raise RuntimeError(f"MultiTalk exited {code}; see {log_path}")

    produced = Path(f"{out_stem}.mp4")
    if not produced.exists():
        raise RuntimeError(f"MultiTalk finished but no output at {produced}; see {log_path}")
    return produced

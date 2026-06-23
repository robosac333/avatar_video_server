"""
Adapter for Tencent-Hunyuan/HunyuanVideo-Avatar.

Hunyuan ships a "GPU-poor" sampling script (`hymm_sp/sample_gpu_poor.py`) that
runs at 480p on a single 24GB-ish card, plus a full-quality script for big
GPUs. We target the gpu_poor path by default since the whole point here is
cheap reels. It reads image + audio + a prompt and writes an mp4 to --save-path.

Reference: https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar
"""
import sys
from pathlib import Path

from .. import config
from ..utils import run_logged


def generate(job_dir: Path, image_path: Path, audio_path: Path, prompt: str,
             resolution: str = "480") -> Path:
    """Run HunyuanVideo-Avatar (gpu-poor mode). Returns path to generated mp4."""
    out_dir = job_dir / "hunyuan_out"
    out_dir.mkdir(exist_ok=True)
    log_path = job_dir / "generate.log"

    # Hunyuan reads a small TSV/CSV describing each sample. One row = one reel.
    samples = job_dir / "samples.csv"
    samples.write_text(
        "videoid,image,audio,prompt\n"
        f"reel,{image_path},{audio_path},{prompt}\n"
    )

    image_size = 704 if resolution == "720" else 512

    cmd = [
        sys.executable, "hymm_sp/sample_gpu_poor.py",
        "--input", str(samples),
        "--ckpt", str(config.HUNYUAN_CKPT),
        "--sample-n-frames", "129",
        "--infer-steps", str(config.SAMPLE_STEPS),
        "--image-size", str(image_size),
        "--save-path", str(out_dir),
        "--use-fp8",            # fp8 weights -> fits the gpu-poor budget
        "--cpu-offload",        # stream layers from CPU when VRAM is tight
        "--infer-min",
    ]

    code = run_logged(
        cmd, cwd=config.HUNYUAN_REPO, log_path=log_path,
        extra_env={"CUDA_VISIBLE_DEVICES": config.CUDA_VISIBLE_DEVICES},
    )
    if code != 0:
        raise RuntimeError(f"HunyuanVideo-Avatar exited {code}; see {log_path}")

    # The script writes one mp4 into out_dir; grab the first.
    mp4s = sorted(out_dir.rglob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"Hunyuan finished but no mp4 under {out_dir}; see {log_path}")
    return mp4s[0]

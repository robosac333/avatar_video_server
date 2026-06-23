# Avatar Lip-Sync Server (self-hosted on RunPod)

Give it an **image + audio + prompt**, get back a **lip-synced talking video**.
Runs two open-weight models on your own GPU so the marginal cost per reel drops
to roughly the GPU's per-second rent (₹15–35) instead of a per-API fee:

- **Wan 2.1 MultiTalk** — your everyday driver. Lighter, faster, great lips.
- **HunyuanVideo-Avatar** — heavier, higher-fidelity "hero" content.

A FastAPI server keeps things warm and serialises jobs onto the single GPU; a
Streamlit UI sits on top for click-and-render.

---

## How the pieces fit

```
            ┌──────────────┐     image+audio+prompt    ┌───────────────────┐
  you  ───► │ Streamlit UI │ ────────────────────────► │  FastAPI server   │
            │  (port 8501) │ ◄──── status / mp4 ─────── │   (port 8000)     │
            └──────────────┘                            │  1 worker thread  │
                                                        │        │          │
                                                        │   ┌────▼─────┐    │
                                                        │   │ adapter  │    │ subprocess
                                                        │   │ multitalk│────┼──► generate_multitalk.py
                                                        │   │  hunyuan │────┼──► sample_gpu_poor.py
                                                        │   └──────────┘    │
                                                        └───────────────────┘
                                                          weights + repos on
                                                          /workspace volume
```

**Why one worker thread?** Two diffusion jobs on one GPU just fight over VRAM
and both finish slower. Serialising is faster and simpler. When you outgrow one
GPU, run a second pod rather than threading harder.

---

## Files

```
avatar-video-server/
├── server/
│   ├── app.py                  FastAPI: /generate /status /result /healthz + queue
│   ├── config.py               every path & tunable (env-overridable)
│   ├── utils.py                audio probing, image prep, logged subprocess
│   └── adapters/
│       ├── multitalk_adapter.py   builds input.json, runs generate_multitalk.py
│       └── hunyuan_adapter.py     runs sample_gpu_poor.py
├── ui/streamlit_app.py         the click-and-render UI
├── scripts/
│   ├── download_models.sh      pull all weights onto the volume (run once)
│   ├── setup_runpod.sh         clone repos, install deps, wire weights (run once)
│   └── run.sh                  start server + UI together
├── Dockerfile                  optional reproducible image
├── requirements.txt            light API/UI deps only
└── .env.example                tunables
```

---

## RunPod setup, step by step

You signed up via GitHub but have **no credits** — so the very first step is
billing. Nothing below runs without a GPU attached.

### 1. Add credits
RunPod → **Billing** → add credit (a few dollars is plenty to test). On-demand
pods bill **per second**, so a handful of test reels costs cents.

### 2. Create a Network Volume
RunPod → **Storage** → **Network Volume** → ~120 GB, in a region that has the
GPU you want. This holds the ~85 GB of weights so you don't re-download them
every pod start. **This is the single most important cost-saver.**

### 3. Deploy a Pod
- **Template:** `RunPod PyTorch 2.4` (or point at your own image built from the
  Dockerfile).
- **GPU:**
  - **RTX 4090 / L40S (24–48 GB)** — fine for MultiTalk 480p with the low-VRAM
    defaults. Cheapest sensible choice.
  - **A100 80GB** — comfortable for both models and 720p; raise
    `NUM_PERSISTENT_PARAM_IN_DIT` to use the headroom for speed.
- **Attach** the Network Volume at `/workspace`.
- **Expose HTTP ports** `8000` and `8501`.

### 4. One-time install (in the pod's web terminal)
```bash
cd /workspace
git clone <your-fork-or-upload-this-repo> avatar-video-server
cd avatar-video-server

pip install --break-system-packages -r requirements.txt
bash scripts/download_models.sh      # ~85 GB, once per volume — go get coffee
bash scripts/setup_runpod.sh         # clones model repos, installs ML deps
```

### 5. Run
```bash
bash scripts/run.sh
```
Open the pod's proxy URL for port **8501** → the Studio UI. Upload Rohini's
portrait, the ElevenLabs audio, tweak the prompt, hit **Generate**.

Check `8000/healthz` first if anything looks off — it tells you whether each
model's weights are actually present on the volume.

---

## Speed & cost knobs (in `.env` / `config.py`)

| Knob | Faster / cheaper | Better quality |
|---|---|---|
| `DEFAULT_RESOLUTION` | `480` | `720` |
| `SAMPLE_STEPS` | `20` | `40` |
| `ENABLE_TEACACHE` | `true` | `false` |
| `NUM_PERSISTENT_PARAM_IN_DIT` | `0` (low VRAM) | large number (needs big GPU) |

Practical recipe for reels: **480p, 25 steps, TeaCache on**. Render 720p only
for a final cut you're happy with. With FlashAttention 2 installed (setup script
attempts it) a ~40s reel on a 4090 lands in a couple of minutes; on an A100 with
params kept resident, well under that.

**Stop the pod when idle.** On-demand billing is per-second only while running;
a stopped pod keeps the volume (small storage fee) and costs no GPU time. Start
it again when you batch the next set of reels.

---

## Calling the API directly (for n8n later)

The whole point of the server boundary is that your n8n automation can hit it
like any web service — no UI needed:

```bash
# submit
curl -s -X POST http://POD:8000/generate \
  -F image=@rohini.jpg -F audio=@line.mp3 \
  -F 'prompt=warm engaged presenter, natural head movements' \
  -F model=multitalk -F resolution=480
# -> {"job_id":"abc123...","audio_seconds":38.0,"queue_position":1}

# poll
curl -s http://POD:8000/status/abc123

# download when status == done
curl -s http://POD:8000/result/abc123 -o reel.mp4
```

That maps cleanly onto your planned pipeline: Sheets → ElevenLabs → **this
endpoint** → captions → Instagram.

---

## Honest caveats

- **Weights & scripts are upstream.** The adapters call each repo's official
  generation script with its documented flags. If a repo changes a flag name,
  you adjust one line in that adapter — the server, queue, and UI are untouched.
  Verify the exact flags against the current repo READMes the first time you set
  up (`generate_multitalk.py --help`, Hunyuan's sample script).
- **First render after a cold start is slowest** (model loads into VRAM). Keep
  the pod running across a batch so subsequent reels are quick.
- **VRAM reality:** if MultiTalk OOMs, keep `NUM_PERSISTENT_PARAM_IN_DIT=0` and
  stay at 480p; if Hunyuan OOMs on a 24 GB card, it's expected — that one really
  wants 48–80 GB for comfort. Use MultiTalk as the workhorse and Hunyuan only
  when you want a showcase clip on a bigger GPU.
- **Licenses:** both models are open-weight but have their own terms — check
  each repo's license before commercial use.

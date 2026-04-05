# Police Scanner – Live Audio, Clips & Speech-to-Text

A locally-hosted web application for your Uniden police scanner.  
Streams live audio to your browser, saves every transmission as a clip, and transcribes each one with GPU-accelerated Whisper.  Clips are searchable by keyword and browsable by date.

---

## Features

- **Live audio** – scanner audio streams to your browser in near-realtime via WebSocket
- **Spectrum visualiser** – frequency display shows when a transmission is active
- **Auto clip saving** – Voice Activity Detection captures each transmission as a `.wav` file
- **GPU speech-to-text** – OpenAI Whisper `large-v3-turbo` on CUDA for maximum accuracy
- **Full-text search** – search all transcripts instantly (SQLite FTS5)
- **Date browser** – browse clips from any past day
- **Real-time updates** – new clips and transcripts appear live without refreshing the page

---

## Requirements

| Hardware | Minimum |
|---|---|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) or better |
| CPU | Any modern x86-64 (i7-12650H or equivalent) |
| Audio | Uniden scanner → laptop mic/line-in jack |
| OS | Ubuntu 22.04+ / Debian 12+ (Windows via WSL2 also works) |

| Software | Version |
|---|---|
| Python | 3.10, 3.11, or 3.12 |
| NVIDIA driver | ≥ 525 (ships with CUDA 12) |
| Browser | Chrome, Edge, or Firefox (Web Audio API + AudioWorklet required) |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/evilgenius79/Online-police-scanner-with-speech-to-text
cd Online-police-scanner-with-speech-to-text

# 2. Run the setup script
#    Creates a venv, installs all packages, installs CUDA libs, makes directories
bash setup.sh

# 3. Find your scanner's audio device index
source venv/bin/activate
python list_devices.py
```

Example output from `list_devices.py`:
```
Available audio INPUT devices:

   IDX   SAMPLERATE   CH  NAME
  ──────────────────────────────────────────────────────────────────────────
  [  0]     44100 Hz    2  Built-in Microphone
  [  1]     48000 Hz    2  USB Audio Device         <-- scanner plugged in here
  [  2]     44100 Hz    2  HD Webcam Microphone
```

```bash
# 4. Edit config.py – set your device index
#    Change:  AUDIO_DEVICE_INDEX = None
#    To:      AUDIO_DEVICE_INDEX = 1   (or whatever index your scanner shows as)
nano config.py

# 5. Start the server
python run.py
```

Open **http://localhost:8000** in your browser.  
Click **Connect Audio** – you should hear the scanner live.

---

## Configuration

All settings are in `config.py`.  The most important ones:

```python
# ── Input device ──────────────────────────────────────────────────────────────
AUDIO_DEVICE_INDEX = None    # None = system default; set to int from list_devices.py

# ── Speech-to-text model ──────────────────────────────────────────────────────
WHISPER_MODEL = "large-v3-turbo"   # best accuracy/speed balance on RTX 4060
# WHISPER_MODEL = "medium.en"      # faster, less accurate
# WHISPER_MODEL = "large-v3"       # maximum accuracy, needs full 8 GB VRAM

# ── VAD sensitivity ───────────────────────────────────────────────────────────
VAD_AGGRESSIVENESS = 2       # 0 (least) – 3 (most aggressive noise rejection)
SILENCE_FRAMES_END = 67      # frames of silence (×30 ms) before clip ends (~2 s)
```

### VAD tuning

| Problem | Fix |
|---|---|
| Clips not triggering / cutting off early | Lower `VAD_AGGRESSIVENESS` (try `1` or `0`), or lower `ENERGY_SILENCE` in `audio_capture.py` (line ~227) |
| Too many false-positive clips | Raise `VAD_AGGRESSIVENESS` to `3`, or raise `ENERGY_SILENCE` to `500`+ |
| Clips split mid-sentence | Raise `SILENCE_FRAMES_END` (e.g. `100` = ~3 s) |
| Clips include too much dead air | Lower `SILENCE_FRAMES_END` |

### CPU fallback (no GPU)

```python
# In config.py:
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_MODEL        = "small.en"   # large-v3-turbo is too slow on CPU
```

---

## Project Structure

```
.
├── config.py               ← all settings
├── run.py                  ← start the server
├── list_devices.py         ← find your mic input index
├── setup.sh                ← automated install script
│
├── app/
│   ├── audio_capture.py    ← sounddevice capture + VAD state machine + clip saving
│   ├── transcriber.py      ← faster-whisper GPU worker thread
│   ├── broadcaster.py      ← thread→asyncio bridge for WebSocket broadcast
│   ├── database.py         ← SQLite + FTS5 full-text search
│   └── main.py             ← FastAPI routes, WebSocket endpoint, lifecycle
│
├── templates/
│   └── index.html          ← single-page web UI
│
├── static/
│   ├── css/style.css       ← dark scanner theme
│   ├── js/app.js           ← live audio, visualiser, clips browser, search
│   └── js/pcm-processor.js ← AudioWorklet (runs on browser audio thread)
│
├── clips/                  ← saved WAV files, organised by date
│   └── YYYY-MM-DD/
│       └── HHMMSS_ffffff.wav
│
├── models/                 ← Whisper model cache (downloaded on first run)
└── scanner.db              ← SQLite database (created on first run)
```

---

## How It Works

```
Uniden scanner
     │  audio via mic/line-in
     ▼
sounddevice InputStream (16 kHz, mono, 30 ms frames)
     │
     ├──► AudioBroadcaster ──► WebSocket ──► Browser AudioWorklet ──► speakers
     │
     └──► VAD worker thread
              │  energy pre-filter + webrtcvad
              │  detects transmission start/end
              │
              ├── saves WAV to  clips/YYYY-MM-DD/HHMMSS.wav
              ├── inserts row in  scanner.db
              └── queues clip for transcription
                       │
                       ▼
              faster-whisper (GPU, float16)
                       │
                       ├── updates transcript in  scanner.db
                       └── broadcasts transcript via WebSocket to browser
```

### Audio pipeline details

- **Sample rate:** 16 000 Hz mono (optimal for both VAD and Whisper)
- **Frame size:** 30 ms → 480 samples → 960 bytes (required by webrtcvad)
- **VAD:** RMS energy pre-filter → webrtcvad confirmation.  Scanner squelch gives clean transitions so this is highly reliable.
- **Pre-roll:** 0.5 s of audio before VAD triggers is prepended to every clip so the first syllable is never cut off
- **Post-roll:** 0.3 s of silence is kept after the last speech frame
- **Live stream:** raw 16-bit PCM is sent as binary WebSocket frames → browser AudioWorklet converts Int16 → Float32 and plays continuously

### Whisper configuration

```python
model.transcribe(
    audio,
    language            = "en",   # skip language detection
    beam_size           = 5,
    vad_filter          = True,    # Whisper's built-in silence removal
    temperature         = 0,       # deterministic; prevents hallucination on silence
    condition_on_previous_text = False,  # clips are independent
    initial_prompt      = "Police radio scanner. Ten-four. Unit responding. Over.",
)
```

The `initial_prompt` primes Whisper with police-radio vocabulary, improving accuracy on codes, phonetic alphabet, and clipped transmissions.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `WS` | `/ws/audio` | Live PCM audio + JSON events |
| `GET` | `/api/clips` | Paginated clip list (`?page=1&per_page=20&date=YYYY-MM-DD`) |
| `GET` | `/api/clips/{id}` | Single clip |
| `GET` | `/api/search` | Full-text search (`?q=ambulance&page=1`) |
| `GET` | `/api/dates` | All dates with clip counts |
| `GET` | `/api/status` | Server status (recording, model loaded, etc.) |
| `GET` | `/api/devices` | List audio input devices |
| `GET` | `/audio/{date}/{filename}` | Serve audio clip file |

### WebSocket event types

Events arrive as JSON text frames over `/ws/audio` (binary frames are PCM audio):

```json
{ "type": "transmission_start", "timestamp": "2024-01-15T14:30:22.123" }
{ "type": "transmission_end",   "timestamp": "2024-01-15T14:30:28.456" }
{ "type": "new_clip",  "clip": { "id": 42, "filename": "...", "duration": 6.3, ... } }
{ "type": "transcript_ready", "clip_id": 42, "transcript": "Unit 12 en route." }
{ "type": "ping" }
```

---

## Accessing from Other Devices

The server binds to `0.0.0.0` by default, so any device on the same LAN can connect:

```
http://<laptop-ip>:8000
```

Find your laptop's IP with `ip addr show` or `hostname -I`.

To restrict to localhost only, edit `config.py`:
```python
HOST = "127.0.0.1"
```

---

## Disk Usage

Clips accumulate over time.  A rough estimate:

- 16 kHz 16-bit mono WAV: **~32 KB/second** of transmission
- Typical scanner activity: 5–20 minutes of transmissions per hour
- Daily usage: **~600 MB – 2.4 GB/day** at moderate activity

The `clips/` directory is organised by date (`clips/YYYY-MM-DD/`) for easy manual cleanup.  To delete clips older than 30 days:

```bash
find clips/ -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
```

---

## Troubleshooting

**No audio captured / scanner is silent in the browser**
- Run `python list_devices.py` and confirm `AUDIO_DEVICE_INDEX` points to the correct input
- Check your OS volume mixer — the line-in level may be muted
- Try turning the scanner volume up; very quiet input = low RMS = VAD won't trigger

**Transcription says "No speech detected" on every clip**
- The clip contains only carrier noise or CTCSS tones.  This is normal for some systems.
- Lower `no_speech_threshold` in `transcriber.py` (default `0.6`); try `0.4`

**"CUDA out of memory" error**
- Switch to a smaller model: `WHISPER_MODEL = "medium.en"` in `config.py`
- Or use `WHISPER_COMPUTE_TYPE = "int8_float16"` to reduce VRAM by ~1 GB

**Whisper model download is slow / fails**
- The `large-v3-turbo` model is ~1.5 GB.  It downloads to `models/` on first run.
- If the download fails, delete the partial file in `models/` and restart

**`webrtcvad-wheels` fails to install**
- Try: `pip install webrtcvad` instead (requires `python3-dev` and `gcc`)
- The app falls back to energy-only VAD if webrtcvad is unavailable — it will still work

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Async web server and WebSocket support |
| `sounddevice` | Audio capture from mic/line-in via PortAudio |
| `webrtcvad-wheels` | Google WebRTC Voice Activity Detection |
| `faster-whisper` | GPU-accelerated Whisper inference via CTranslate2 |
| `nvidia-cublas-cu12` | CUDA 12 BLAS library (installed via pip, no system CUDA needed) |
| `nvidia-cudnn-cu12` | cuDNN 9 library (installed via pip) |
| `numpy` | Audio buffer conversion |
| `aiofiles` | Async file serving |

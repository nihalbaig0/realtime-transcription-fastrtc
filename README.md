# Real Time Speech Transcription with FastRTC ⚡️and Local Whisper 🤗 

This project uses FastRTC to handle the live audio streaming and open-source Automatic Speech Recognition models via Transformers.

Check the [FastRTC documentation](https://fastrtc.org/) to see what parameters you can tweak with respect to the audio stream, Voice Activity Detection (VAD), etc.

**System Requirements**
- python >= 3.10
- ffmpeg

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/sofi444/realtime-transcription-fastrtc
cd realtime-transcription-fastrtc
```

### Step 2: Set up environment
Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)


```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```
</details>

<details>
<summary>🐍 Using pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
</details>

### Step 3: Install ffmpeg
<details>
<summary>🍎 macOS</summary>

```bash
brew install ffmpeg
```
</details>

<details>
<summary>🐧 Linux (Ubuntu/Debian)</summary>

```bash
sudo apt update
sudo apt install ffmpeg
```
</details>

### Step 4: Configure environment
Create a `.env` file in the project root:

```env
UI_MODE=fastapi
APP_MODE=local
SERVER_NAME=localhost
```

- **UI_MODE**: controls the interface to use. If set to `gradio`, you will launch the app via Gradio and use their default UI. If set to anything else (eg. `fastapi`) it will use the `index.html` file in the root directory to create the UI (you can customise it as you want) (default `fastapi`).
- **APP_MODE**: ignore this if running only locally. If you're deploying eg. in Spaces, you need to configure a Turn Server. In that case, set it to `deployed`, follow the instructions [here](https://fastrtc.org/deployment/) (default `local`).
- **MODEL_ID**: HF model identifier for the ASR model you want to use (see [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)) (default `openai/whisper-large-v3-turbo`)
- **SERVER_NAME**: Host to bind to (default `localhost`)
- **PORT**: Port number (default `7860`) 

### Step 5: Launch the application
```bash
python main.py
```
click on the url that pops up (eg. https://localhost:7860) to start using the app!


### Whisper

Choose the Whisper model version you want to use. See all [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending&search=whisper) - you can of course also use a non-Whisper ASR model.

On MPS, I can run `whisper-large-v3-turbo` without problems. This is my current favourite as it’s lightweight, performant and multi-lingual!

Adjust the parameters as you like, but remember that for real-time, we want the batch size to be 1 (i.e. start transcribing as soon as a chunk is available).

If you want to transcribe different languages, set the language parameter to the target language, otherwise Whisper defaults to translating to English (even if you set `transcribe` as the task).

## Docker 🐳 (optional)

I provide a Docker setup for both CPU and GPU: `Dockerfile.cpu` and `Dockerfile.cuda`, helpful if you want to deploy the app in a container.

The Dockerfiles use `uv` as environment and package manager.

`Dockerfile.cuda` includes Flash Attention installation for faster inference (https://github.com/Dao-AILab/flash-attention).


<details>
<summary>🖥️ CPU-Only</summary>
For CPU-only inference (works on any system).

Using the `docker-compose.yml` file provided, run:

```bash
docker-compose --profile cpu up --build
```

Otherwise, you can build the image manually:
```bash
docker build -f Dockerfile.cpu -t realtime-transcription-fastrtc-cpu .
```

</details>

<details>
<summary>🚀 GPU Deployment (NVIDIA)</summary>
For GPU-accelerated inference.

Using the `docker-compose.yml` file provided, run:

```bash
docker-compose --profile cuda up --build
```

Otherwise, you can build the image manually:

```bash
docker build -f Dockerfile.cuda -t realtime-transcription-fastrtc-cuda .
```

**NOTE**: Requires NVIDIA GPU with CUDA 12.1. Change base image in `Dockerfile.cuda` to match your CUDA version.
</details>

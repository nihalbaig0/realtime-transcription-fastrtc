# Live Speech Transcription powered by FastRTC ⚡️ and Local Whisper 🤗 

This project uses [FastRTC](https://fastrtc.org/) to handle live interaction between audio input and text output, and open-source Automatic Speech Recognition (ASR) models via [Transformers](https://huggingface.co/docs/transformers).

## Quick Start 🚀

**System Requirements**
- Python ≥3.10
- ffmpeg
- CUDA-compatible GPU (optional, for faster inference)

```bash
# Clone and enter directory
git clone https://github.com/sofi444/realtime-transcription-fastrtc
cd realtime-transcription-fastrtc

# Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run with default settings
python main.py
```

Visit the URL shown in the terminal (default: https://localhost:7860) to start transcribing!

## Detailed Installation

### 1. Environment Setup

Choose your preferred package manager:

<details>
<summary>📦 UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```
</details>

<details>
<summary>🐍 pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
</details>

### 2. Install ffmpeg

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

## Configuration

The application can be configured using environment variables. Create a `.env` file in the project root with the following variables:

| Variable | Description | Possible Values | Default |
|----------|-------------|-----------------|---------|
| `UI_MODE` | Interface type to use | `fastapi` (custom UI), `gradio` (default Gradio UI) | `fastapi` |
| `UI_TYPE` | UI template to use when `UI_MODE=fastapi` | `base`, `screen` | `base` |
| `APP_MODE` | Deployment mode | `local`, `deployed` | `local` |
| `TURN_PROVIDER` | TURN server provider when `APP_MODE=deployed` | `hf-cloudflare`, `cloudflare`, `twilio` | `hf-cloudflare` |
| `MODEL_ID` | HuggingFace model identifier | Any | `openai/whisper-large-v3-turbo` |
| `LANGUAGE` | Target language for transcription | Any valid or None | `english` |
| `SERVER_NAME` | Host to bind to | Any valid hostname | `localhost` |
| `PORT` | Port number | Any valid port | `7860` |

## Model Selection

You can use any Whisper model version or other ASR model from [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending). The default `whisper-large-v3-turbo` is recommended as it's lightweight, performant and multi-lingual.

We use batch size 1 to start transcribing as soon as a chunk is available.

## Docker 🐳

I provide a Docker setup for both CPU and GPU: `Dockerfile.cpu` and `Dockerfile.cuda`, helpful if you want to deploy the app in a container.

The Dockerfiles use `uv` as environment and package manager.

`Dockerfile.cuda` includes Flash Attention installation for faster inference (https://github.com/Dao-AILab/flash-attention).


<details>
<summary>🖥️ CPU-Only</summary>

```bash
# Using docker-compose
docker-compose --profile cpu up --build

# Or build manually
docker build -f Dockerfile.cpu -t realtime-transcription-fastrtc-cpu .
```
</details>

<details>
<summary>🚀 GPU Deployment (NVIDIA)</summary>

```bash
# Using docker-compose
docker-compose --profile cuda up --build

# Or build manually
docker build -f Dockerfile.cuda -t realtime-transcription-fastrtc-cuda .
```

**Note**: Requires NVIDIA GPU with CUDA 12.1. Change base image in `Dockerfile.cuda` to match your CUDA version.
</details>


## Deploying on HF Spaces 🤗

1. Create a new Space on HuggingFace
2. Select Docker SDK
3. Choose your hardware (CPU/GPU)
4. Set variables and secrets in the Space's settings (see [here](https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets))
5. Clone your Space's repository
6. Copy the app contents to the new repository
7. Rename either `Dockerfile.cpu` or `Dockerfile.cuda` to `Dockerfile` based on your hardware choice (see previous section about Docker 🐳)
8. Push to your Space!

For deployed environments:
- Set `APP_MODE=deployed`
- Set `TURN_PROVIDER` to your chosen provider (`cloudflare`, `hf-cloudflare`, or `twilio`)
- Configure the corresponding TURN server credentials in your Space's secrets

## Additional Resources

- [FastRTC documentation](https://fastrtc.org/) - Learn about audio stream and Voice Activity Detection (VAD) parameters
- [Whisper configuration options](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate)
- [Available ASR models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending) on Hugging Face
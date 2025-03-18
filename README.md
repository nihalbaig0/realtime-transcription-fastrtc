# Real Time Speech Transcription with FastRTC ⚡️and Local Whisper 🤗 

--

NOTE! : This is just the PoC code / starting point for a volunteering project I am doing - I wasn't expecting it to get so much attention :O

I am going to improve it as soon as possible! Any contributions would be much appreciated :))

--


## Installation

Step 1: `git clone https://github.com/sofi444/realtime-transcription-fastrtc`

Step 2: `cd realtime-transcription-fastrtc`

Step 3: `uv venv --python 3.11 && source .venv/bin/activate` (you might need to install uv)

Step 4: `uv pip install -r requirements.txt`

Step 5: `brew install ffmpeg` (or your system specific way of installing ffmpeg)

Step 6: `python main.py` (follow along the fun at https://localhost:7860)

Bonus: It's recommended to create a .env file and add the following variables:

```
UI_MODE=fastapi
APP_MODE=x

SERVER_NAME=localhost
```

Enjoy! :))


=======

**System Dependencies:** 
- `ffmpeg`
- python = ">=3.10"


FastRTC docs : https://fastrtc.org/ - check out what parameters you can tweak with respect to the Audio Stream, Voice Activity Detection, etc.

### Whisper

Choose the Whisper model version you want to use, depending on your hardware. See all [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending&search=whisper) - you can of course also use a non-Whisper ASR model.

On MPS, I can run `whisper-large-v3-turbo` without problem. This is my current favourite as it’s lightweight, performant and multi-lingual!

Adjust the parameters as you like, but remember that for real-time, we want the batch size to be 1 (i.e. start transcribing as soon as a chunk is available).

If you want to transcribe different languages, set the language parameter to the target language, otherwise Whisper defaults to translating to English (even if you set `transcribe` as the task).

### Server / UI 

By selecting GRADIO you’ll launch a Gradio server and default UI.

You can also launch a FastAPI server with custom UI.

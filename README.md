# Real Time Speech Transcription with FastRTC ⚡️and Local Whisper 🤗 

`main.py` : transcribe away!

FastRTC docs : https://fastrtc.org/ - check out what parameters you can tweak with respect to the Audio Stream, Voice Activity Detection, etc.

### Whisper

Choose the Whisper model version you want to use, depending on your hardware. See all [here](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending&search=whisper) - you can of course also use a non-Whisper ASR model.

On MPS, I can run `whisper-large-v3-turbo` without problem. This is my current favourite as it’s lightweight, performant and multi-lingual!

Adjust the parameters as you like, but remember that for real-time, we want the batch size to be 1 (i.e. start transcribing as soon as a chunk is available).

If you want to transcribe different languages, set the language parameter to the target language, otherwise Whisper defaults to translating to English (even if you set `transcribe` as the task).

### Server / UI 

By selecting GRADIO you’ll launch a Gradio server and default UI.

You can also launch a FastAPI server with custom UI.

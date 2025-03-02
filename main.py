import os
import logging

import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    AlgoOptions,
    SileroVadOptions,
    get_hf_turn_credentials,
)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available
import sounddevice as sd
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from utils.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    else:
        return "cpu"
    
def get_torch_and_np_dtypes(device, use_bfloat16=False):
    if device == "cuda":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    elif device == "mps":
        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    return torch_dtype, np_dtype

device = get_device(force_cpu=False)
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
logger.info(
    f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}"
)

attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
logger.info(f"Using attention: {attention}")

model_id = "openai/whisper-large-v3-turbo"
logger.info(f"Loading Whisper model: {model_id}")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True,
    attn_implementation=attention
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcribe_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Warm up the model with empty audio
logger.info("Warming up Whisper model with dummy input")
warmup_audio = np.zeros((16000,), dtype=np_dtype)  # 1s of silence
transcribe_pipeline(warmup_audio)
logger.info("Model warmup complete")

async def transcribe(audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    # Convert to mono if stereo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    
    audio_array = audio_array.astype(np_dtype)
    audio_array /= np.max(np.abs(audio_array))
    
    outputs = transcribe_pipeline(
        {"sampling_rate": sample_rate, "raw": audio_array},
        chunk_length_s=3,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
        },
        #return_timestamps="word"
    )
    yield AdditionalOutputs(outputs["text"].strip())

# Get credentials for the community TURN server (for when deployed on Spaces)
try:
    credentials = get_hf_turn_credentials(token=os.getenv("HF_TOKEN"))
except Exception as e:
    logger.error(f"Error getting credentials: {e}")
    credentials = None

logger.info("Initializing FastRTC stream")
stream = Stream(
    handler=ReplyOnPause(
        transcribe,
        #algo_options=AlgoOptions(
            # Duration in seconds of audio chunks (default 0.6)
            #audio_chunk_duration=3,
            # If the chunk has more than started_talking_threshold seconds of speech, the user started talking (default 0.2)
            #started_talking_threshold=0.2,
            # If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking. (default 0.1)
            #speech_threshold=0.1,
        #),
        #model_options=SileroVadOptions(
            # Threshold for what is considered speech (default 0.5)
            #threshold=0.3,
            # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
            #min_speech_duration_ms=200,
            # Max duration of speech chunks, longer will be split (default float('inf'))
            #max_speech_duration_s=30,
            # Wait for ms at the end of each speech chunk before separating it (default 2000)
            #min_silence_duration_ms=2000,
            # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
            #window_size_samples=1024,
            # Final speech chunks are padded by speech_pad_ms each side
            #speech_pad_ms=400,
        #),
    ),
    # send-receive: bidirectional streaming (default)
    # send: client to server only
    # receive: server to client only
    modality="audio",
    mode="send",
    additional_outputs=[
        gr.Textbox(label="Transcript"),
    ],
    additional_outputs_handler=lambda current, new: current + " " + new,
    rtc_configuration=credentials
)

app = FastAPI()
stream.mount(app)

def list_microphones():
    """List available microphones using sounddevice"""
    devices = sd.query_devices()
    return [d['name'] for d in devices if d['max_input_channels'] > 0]

@app.get("/transcript")
def _(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")
    async def output_stream():
        try:
            async for output in stream.output_stream(webrtc_id):
                transcript = output.args[0]
                logger.debug(f"Sending transcript for {webrtc_id}: {transcript[:50]}...")
                yield f"event: output\ndata: {transcript}\n\n"
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {str(e)}")
            raise

    return StreamingResponse(output_stream(), media_type="text/event-stream")

# HTML for uvicorn mode
@app.get("/")
def index():
    logger.debug("Serving index page")
    microphones = list_microphones()
    with open("display.html", "r") as file:
        html_content = file.read()
    # Populate microphone options
    options = ''.join([f'<option value="{i}">{name}</option>' 
                      for i, name in enumerate(microphones)])
    return HTMLResponse(content=html_content.replace("{{ microphones }}", options))

if __name__ == "__main__":
    mode = os.getenv("MODE", "gradio")
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    port = os.getenv("PORT", 7860)
    logger.info(f"Starting application in {mode} mode")
    
    if mode == "gradio":
        logger.info(f"Launchng Gradio UI on port {port}")
        logger.info(f"Available at http://{server_name}:{port}")
        stream.ui.launch(
            server_port=port, 
            server_name=server_name,
            ssl_verify=False,
            debug=True
        )
    else:
        logger.info(f"Launching FastAPI server on port {port}")
        logger.info(f"Available at http://{server_name}:{port}")
        uvicorn.run(app, host=server_name, port=port)
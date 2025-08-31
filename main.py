import os
import logging
import json
import time
from collections import deque
import threading

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    AlgoOptions,
    SileroVadOptions,
    audio_to_bytes,
)

from utils.logger_config import setup_logging
from utils.device import get_device
from utils.turn_server import get_rtc_credentials
from utils.model import initialize_whisper_model

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

UI_MODE = os.getenv("UI_MODE", "fastapi").lower()
UI_TYPE = os.getenv("UI_TYPE", "base").lower()
APP_MODE = os.getenv("APP_MODE", "local").lower()
TURN_PROVIDER = os.getenv("TURN_PROVIDER", "hf-cloudflare")

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
LANGUAGE = os.getenv("LANGUAGE", "english")

logger.info(f"""
    --------------------------------------
    Near-Streaming Configuration:
    - UI_MODE: {UI_MODE}
    - UI_TYPE: {UI_TYPE}
    - APP_MODE: {APP_MODE}
    - MODEL_ID: {MODEL_ID}
    - LANGUAGE: {LANGUAGE}
    - Mode: Aggressive ReplyOnPause for near-real-time
    --------------------------------------
""")

transcribe_pipeline = initialize_whisper_model(
    model_id=MODEL_ID,
    try_compile=True,
    try_use_flash_attention=True,
    device=get_device(force_cpu=False)
)

# Connection-specific state management
connection_states = {}

class StreamingTranscriptionState:
    def __init__(self):
        self.audio_buffer = deque()
        self.last_transcript = ""
        self.total_transcript = ""
        self.chunk_count = 0
        self.start_time = time.time()
        
    def add_audio(self, audio_chunk):
        self.audio_buffer.append(audio_chunk)
        self.chunk_count += 1
        
        # Limit buffer to prevent memory issues (keep last 30 seconds)
        max_chunks = 30 * 16000 // len(audio_chunk)  # Approximate
        while len(self.audio_buffer) > max_chunks:
            self.audio_buffer.popleft()

async def streaming_transcribe(audio: tuple[int, np.ndarray], connection_id=None):
    """Enhanced transcription with streaming optimizations"""
    sample_rate, audio_array = audio
    
    # Get or create connection state
    if connection_id not in connection_states:
        connection_states[connection_id] = StreamingTranscriptionState()
    
    state = connection_states[connection_id]
    state.add_audio(audio_array)
    
    logger.debug(f"Processing audio chunk {state.chunk_count} for connection {connection_id}")
    logger.debug(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    try:
        # Use shorter chunks for faster processing
        outputs = transcribe_pipeline(
            audio_to_bytes(audio),
            chunk_length_s=3,  # Shorter chunks for faster response
            batch_size=1,      # Process immediately
            generate_kwargs={
                'task': 'transcribe',
                'language': LANGUAGE,
                'no_repeat_ngram_size': 2,  # Reduce repetition
                'temperature': 0.0,         # More deterministic
            },
        )
        
        current_text = outputs["text"].strip()
        
        if current_text and current_text != state.last_transcript:
            # Calculate incremental update
            if state.last_transcript and current_text.startswith(state.last_transcript):
                # Extract new portion
                new_portion = current_text[len(state.last_transcript):].strip()
                if new_portion:
                    state.last_transcript = current_text
                    state.total_transcript += " " + new_portion
                    logger.debug(f"Incremental update: {new_portion}")
                    yield AdditionalOutputs(new_portion)
            else:
                # Complete replacement or first transcription
                state.last_transcript = current_text
                if not state.total_transcript:  # First transcription
                    state.total_transcript = current_text
                    logger.debug(f"First transcription: {current_text}")
                    yield AdditionalOutputs(current_text)
                else:
                    # Handle significant changes
                    state.total_transcript += " " + current_text
                    logger.debug(f"New segment: {current_text}")
                    yield AdditionalOutputs(current_text)
        
    except Exception as e:
        logger.error(f"Transcription error for connection {connection_id}: {e}")

def cleanup_connection(connection_id):
    """Clean up connection state"""
    if connection_id in connection_states:
        logger.info(f"Cleaning up connection {connection_id}")
        del connection_states[connection_id]

# Create FastRTC stream with very aggressive settings for near-real-time
logger.info("Initializing FastRTC stream with aggressive settings for near-streaming")
stream = Stream(
    handler=ReplyOnPause(
        streaming_transcribe,
        algo_options=AlgoOptions(
            # Very short audio chunks for responsiveness
            audio_chunk_duration=0.2,  # 200ms chunks - very responsive
            # Very low thresholds for quick detection
            started_talking_threshold=0.05,  # Detect speech quickly
            speech_threshold=0.05,           # Detect pauses quickly
            # Short max speech before forcing processing
            max_continuous_speech_s=2.0     # Process every 2 seconds max
        ),
        model_options=SileroVadOptions(
            # Aggressive VAD settings for responsiveness
            threshold=0.25,                  # Very sensitive speech detection
            min_speech_duration_ms=50,       # Accept very short speech
            max_speech_duration_s=2.0,       # Force processing frequently
            min_silence_duration_ms=100,     # Very short pause detection
            window_size_samples=512,         # Small window for fast processing
            speech_pad_ms=50,               # Minimal padding for speed
        ),
    ),
    modality="audio",
    mode="send",
    additional_outputs=[
        gr.Textbox(label="Live Transcript"),
    ],
    # Accumulate results for continuous transcript
    additional_outputs_handler=lambda current, new: (current + " " + new).strip() if current else new,
    rtc_configuration=get_rtc_credentials(provider=TURN_PROVIDER) if APP_MODE == "deployed" else None,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
stream.mount(app)

@app.get("/")
async def index():
    if UI_TYPE == "base":
        html_content = open("static/index.html").read()
    elif UI_TYPE == "screen":
        html_content = open("static/index-screen.html").read()

    rtc_configuration = await get_rtc_credentials(provider=TURN_PROVIDER) if APP_MODE == "deployed" else None
    logger.info(f"RTC configuration: {rtc_configuration}")
    html_content = html_content.replace("__INJECTED_RTC_CONFIG__", json.dumps(rtc_configuration))
    return HTMLResponse(content=html_content)

@app.get("/transcript")
def transcript_endpoint(webrtc_id: str):
    logger.debug(f"New transcript stream request for webrtc_id: {webrtc_id}")
    
    async def streaming_output():
        try:
            async for output in stream.output_stream(webrtc_id):
                transcript = output.args[0]
                logger.debug(f"Streaming transcript for {webrtc_id}: {transcript[:50]}...")
                # Use 'partial' event type to indicate streaming nature
                yield f"event: partial\ndata: {transcript}\n\n"
        except Exception as e:
            logger.error(f"Error in transcript stream for {webrtc_id}: {str(e)}")
            cleanup_connection(webrtc_id)
            raise
        finally:
            cleanup_connection(webrtc_id)

    return StreamingResponse(streaming_output(), media_type="text/event-stream")

@app.post("/cleanup/{webrtc_id}")
async def cleanup_connection_endpoint(webrtc_id: str):
    cleanup_connection(webrtc_id)
    return {"status": "cleaned"}

if __name__ == "__main__":
    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)
    
    if UI_MODE == "gradio":
        logger.info("Launching Gradio UI with near-streaming support")
        stream.ui.launch(
            server_port=port, 
            server_name=server_name,
            ssl_verify=False,
            debug=True
        )
    else:
        import uvicorn
        logger.info("Launching FastAPI server with near-streaming support")
        uvicorn.run(app, host=server_name, port=port)

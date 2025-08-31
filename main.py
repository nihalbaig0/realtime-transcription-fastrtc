import os
import logging
import json
import asyncio
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

# Streaming configuration
STREAMING_CHUNK_LENGTH = 2.0  # Process every 2 seconds of audio
OVERLAP_LENGTH = 0.5  # 500ms overlap for better continuity
MAX_BUFFER_LENGTH = 30.0  # Maximum audio buffer length in seconds

logger.info(f"""
    --------------------------------------
    Streaming Configuration:
    - UI_MODE: {UI_MODE}
    - UI_TYPE: {UI_TYPE}
    - APP_MODE: {APP_MODE}
    - MODEL_ID: {MODEL_ID}
    - LANGUAGE: {LANGUAGE}
    - STREAMING_CHUNK_LENGTH: {STREAMING_CHUNK_LENGTH}s
    - OVERLAP_LENGTH: {OVERLAP_LENGTH}s
    - MAX_BUFFER_LENGTH: {MAX_BUFFER_LENGTH}s
    --------------------------------------
""")

transcribe_pipeline = initialize_whisper_model(
    model_id=MODEL_ID,
    try_compile=True,
    try_use_flash_attention=True,
    device=get_device(force_cpu=False)
)

class StreamingTranscriber:
    def __init__(self, pipeline, sample_rate=16000):
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.is_active = False
        self.last_transcript = ""
        self.processing_thread = None
        
    async def add_audio_chunk(self, audio_chunk: np.ndarray):
        """Add new audio chunk to buffer and trigger processing"""
        with self.buffer_lock:
            self.audio_buffer.append(audio_chunk)
            
            # Limit buffer size to prevent memory issues
            total_length = sum(len(chunk) for chunk in self.audio_buffer)
            max_samples = int(MAX_BUFFER_LENGTH * self.sample_rate)
            
            while total_length > max_samples and len(self.audio_buffer) > 1:
                removed_chunk = self.audio_buffer.popleft()
                total_length -= len(removed_chunk)
    
    def get_audio_for_transcription(self):
        """Get current audio buffer for transcription"""
        with self.buffer_lock:
            if not self.audio_buffer:
                return None
                
            # Concatenate all audio chunks
            full_audio = np.concatenate(list(self.audio_buffer))
            
            # Only process if we have enough audio
            min_samples = int(STREAMING_CHUNK_LENGTH * self.sample_rate)
            if len(full_audio) < min_samples:
                return None
                
            return full_audio
    
    async def transcribe_streaming(self):
        """Continuously transcribe available audio"""
        audio = self.get_audio_for_transcription()
        if audio is None:
            return None
            
        try:
            # Use shorter chunks for streaming
            result = self.pipeline(
                audio,
                chunk_length_s=STREAMING_CHUNK_LENGTH,
                batch_size=1,
                generate_kwargs={
                    'task': 'transcribe',
                    'language': LANGUAGE,
                },
                return_timestamps=True  # Enable timestamps for better streaming
            )
            
            # Extract the text
            current_transcript = result.get("text", "").strip()
            
            # Only return if transcript changed significantly
            if current_transcript and current_transcript != self.last_transcript:
                # Calculate the new portion (incremental update)
                if self.last_transcript and current_transcript.startswith(self.last_transcript):
                    new_text = current_transcript[len(self.last_transcript):].strip()
                    if new_text:
                        self.last_transcript = current_transcript
                        return new_text
                else:
                    # Complete replacement
                    self.last_transcript = current_transcript
                    return current_transcript
                    
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            
        return None

# Global transcriber instance per connection
active_transcribers = {}

async def streaming_transcribe_handler(audio: tuple[int, np.ndarray], webrtc_id: str):
    """Handle streaming transcription for continuous audio chunks"""
    sample_rate, audio_array = audio
    
    # Get or create transcriber for this connection
    if webrtc_id not in active_transcribers:
        active_transcribers[webrtc_id] = StreamingTranscriber(transcribe_pipeline, sample_rate)
    
    transcriber = active_transcribers[webrtc_id]
    
    # Add audio chunk to buffer
    await transcriber.add_audio_chunk(audio_array)
    
    # Transcribe current buffer
    result = await transcriber.transcribe_streaming()
    
    if result:
        logger.debug(f"Streaming result for {webrtc_id}: {result[:50]}...")
        yield AdditionalOutputs(result)

class StreamingHandler:
    """Custom handler for continuous streaming transcription"""
    
    def __init__(self, transcribe_func):
        self.transcribe_func = transcribe_func
        self.connections = {}
        
    async def __call__(self, audio, webrtc_id=None):
        # Store connection context
        if webrtc_id and webrtc_id not in self.connections:
            self.connections[webrtc_id] = {
                'start_time': time.time(),
                'chunk_count': 0
            }
        
        if webrtc_id:
            self.connections[webrtc_id]['chunk_count'] += 1
        
        # Process with connection-specific context
        async for result in self.transcribe_func(audio, webrtc_id):
            yield result
    
    def cleanup_connection(self, webrtc_id):
        """Clean up resources for a connection"""
        if webrtc_id in active_transcribers:
            del active_transcribers[webrtc_id]
        if webrtc_id in self.connections:
            del self.connections[webrtc_id]

# Create streaming handler
streaming_handler = StreamingHandler(streaming_transcribe_handler)

logger.info("Initializing FastRTC stream for continuous transcription")
stream = Stream(
    handler=streaming_handler,
    # More aggressive settings for true streaming
    algo_options=AlgoOptions(
        # Smaller chunks for more responsive streaming
        audio_chunk_duration=0.25,  # 250ms chunks
        # Lower thresholds for faster response
        started_talking_threshold=0.05,
        speech_threshold=0.05,
        # Process continuously, don't wait for long pauses
        max_continuous_speech_s=2.0  # Process every 2 seconds max
    ),
    model_options=SileroVadOptions(
        threshold=0.3,  # Lower threshold for more sensitive detection
        min_speech_duration_ms=100,  # Shorter minimum speech
        max_speech_duration_s=2.0,  # Shorter maximum for streaming
        min_silence_duration_ms=200,  # Shorter silence detection
        window_size_samples=512,  # Smaller window for faster processing
        speech_pad_ms=100,  # Less padding for responsiveness
    ),
    modality="audio",
    mode="send",
    additional_outputs=[
        gr.Textbox(label="Live Transcript"),
    ],
    # Accumulate streaming results instead of replacing
    additional_outputs_handler=lambda current, new: (current + " " + new).strip(),
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
    logger.debug(f"New streaming transcript request for webrtc_id: {webrtc_id}")
    
    async def streaming_output():
        try:
            async for output in stream.output_stream(webrtc_id):
                transcript = output.args[0]
                # Send incremental updates
                logger.debug(f"Streaming update for {webrtc_id}: {transcript[:30]}...")
                yield f"event: streaming\ndata: {transcript}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming transcript for {webrtc_id}: {str(e)}")
            # Clean up resources on error
            streaming_handler.cleanup_connection(webrtc_id)
            raise
        finally:
            # Clean up when stream ends
            streaming_handler.cleanup_connection(webrtc_id)

    return StreamingResponse(streaming_output(), media_type="text/event-stream")

# Cleanup endpoint for graceful connection termination
@app.post("/cleanup/{webrtc_id}")
async def cleanup_connection(webrtc_id: str):
    streaming_handler.cleanup_connection(webrtc_id)
    return {"status": "cleaned"}

if __name__ == "__main__":
    server_name = os.getenv("SERVER_NAME", "localhost")
    port = os.getenv("PORT", 7860)
    
    if UI_MODE == "gradio":
        logger.info("Launching Gradio UI with streaming support")
        stream.ui.launch(
            server_port=port, 
            server_name=server_name,
            ssl_verify=False,
            debug=True
        )
    else:
        import uvicorn
        logger.info("Launching FastAPI server with streaming support")
        uvicorn.run(app, host=server_name, port=port)

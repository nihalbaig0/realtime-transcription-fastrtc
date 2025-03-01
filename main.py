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

from utils.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

device, torch_dtype = get_device()
logger.info(f"Using device: {device}, torch_dtype: {torch_dtype}")

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
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Warm up the model with empty audio
#logger.info("Warming up Whisper model with dummy input")
#warmup_audio = np.zeros((16000,), dtype=np.float32)  # 1s of silence
#transcribe_pipeline(warmup_audio)
#logger.info("Model warmup complete")

async def transcribe(audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.debug(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    # Convert to mono if needed
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    audio_array = audio_array.astype(np.float32)
    
    outputs = transcribe_pipeline(
        audio_array,
        chunk_length_s=30,
        batch_size=24,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
            'return_timestamps': True
        },
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
    handler=ReplyOnPause(transcribe),
    modality="audio",
    mode="send",
    additional_outputs=[
        gr.Textbox(label="Transcript"),
    ],
    additional_outputs_handler=lambda current, new: current + " " + new if current else new,
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

# HTML for fastapi mode
@app.get("/")
def index():
    logger.debug("Serving index page")
    microphones = list_microphones()
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Speech Recognition</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #transcript { 
                width: 100%; 
                height: 200px; 
                margin: 20px 0; 
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover { background-color: #45a049; }
            #micSelection { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Real-time Speech Recognition</h1>
        <div id="micSelection">
            <label for="micList">Select Microphone:</label>
            <select id="micList">
                {{ microphones }}
            </select>
        </div>
        <button id="startButton">Start Recording</button>
        <button id="stopButton" disabled>Stop Recording</button>
        <div id="transcript" contenteditable="true"></div>

        <script>
            let pc;
            let dataChannel;

            async function setupWebRTC() {
                const selectedMic = document.getElementById('micList').value;
                pc = new RTCPeerConnection();
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { deviceId: selectedMic ? { exact: selectedMic } : true }
                });
                
                stream.getTracks().forEach(track => {
                    pc.addTrack(track, stream);
                });

                dataChannel = pc.createDataChannel("text");
                const webrtc_id = Math.random().toString(36).substring(7);

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                const response = await fetch('/webrtc/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: offer.sdp,
                        type: offer.type,
                        webrtc_id: webrtc_id
                    })
                });

                const answer = await response.json();
                await pc.setRemoteDescription(answer);

                // Setup SSE for transcripts
                const eventSource = new EventSource('/transcript?webrtc_id=' + webrtc_id);
                eventSource.addEventListener('output', (event) => {
                    const transcript = document.getElementById('transcript');
                    transcript.textContent = event.data;
                });

                return pc;
            }

            function stopWebRTC(pc) {
                if (pc) {
                    pc.getTransceivers().forEach(transceiver => {
                        if (transceiver.stop) {
                            transceiver.stop();
                        }
                    });
                    pc.getSenders().forEach(sender => {
                        if (sender.track) sender.track.stop();
                    });
                    setTimeout(() => pc.close(), 500);
                }
            }

            document.getElementById('startButton').onclick = async () => {
                pc = await setupWebRTC();
                document.getElementById('startButton').disabled = true;
                document.getElementById('stopButton').disabled = false;
            };

            document.getElementById('stopButton').onclick = () => {
                stopWebRTC(pc);
                document.getElementById('startButton').disabled = false;
                document.getElementById('stopButton').disabled = true;
            };
        </script>
    </body>
    </html>
    """
    # Populate microphone options
    options = ''.join([f'<option value="{i}">{name}</option>' 
                      for i, name in enumerate(microphones)])
    return HTMLResponse(content=html_content.replace("{{ microphones }}", options))

if __name__ == "__main__":
    mode = os.getenv("MODE", "gradio")
    logger.info(f"Starting application in {mode} mode")
    
    if mode == "gradio":
        logger.info("Launching Gradio UI on port 7860")
        stream.ui.launch(server_port=7860, server_name=os.getenv("SERVER_NAME", "0.0.0.0"))
    elif mode == "fastapi":
        logger.info("Launching FastAPI server on port 7860")
        uvicorn.run(app, host=os.getenv("SERVER_NAME", "0.0.0.0"), port=7860)
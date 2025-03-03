import os
import logging

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    AlgoOptions,
    SileroVadOptions,
    get_hf_turn_credentials,
    audio_to_bytes,
)
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.logger_config import setup_logging
from utils.device import get_device, get_torch_and_np_dtypes


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


device = get_device(force_cpu=False)
torch_dtype, np_dtype = get_torch_and_np_dtypes(device, use_bfloat16=False)
logger.info(f"Using device: {device}, torch_dtype: {torch_dtype}, np_dtype: {np_dtype}")


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


# Get credentials for the community TURN server (for when deployed on Spaces)
try:
    credentials = get_hf_turn_credentials(token=os.getenv("HF_TOKEN"))
except Exception as e:
    logger.error(f"Error getting credentials: {e}")
    credentials = None


async def transcribe(audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    outputs = transcribe_pipeline(
        audio_to_bytes(audio),
        chunk_length_s=3,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
        },
        #return_timestamps="word"
    )
    yield AdditionalOutputs(outputs["text"].strip())


logger.info("Initializing FastRTC stream")
stream = Stream(
    handler=ReplyOnPause(
        transcribe,
        algo_options=AlgoOptions(
            # Duration in seconds of audio chunks (default 0.6)
            audio_chunk_duration=0.6,
            # If the chunk has more than started_talking_threshold seconds of speech, the user started talking (default 0.2)
            started_talking_threshold=0.2,
            # If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking. (default 0.1)
            speech_threshold=0.1,
        ),
        model_options=SileroVadOptions(
            # Threshold for what is considered speech (default 0.5)
            threshold=0.5,
            # Final speech chunks shorter min_speech_duration_ms are thrown out (default 250)
            min_speech_duration_ms=250,
            # Max duration of speech chunks, longer will be split (default float('inf'))
            max_speech_duration_s=30,
            # Wait for ms at the end of each speech chunk before separating it (default 2000)
            min_silence_duration_ms=2000,
            # Chunk size for VAD model. Can be 512, 1024, 1536 for 16k s.r. (default 1024)
            window_size_samples=1024,
            # Final speech chunks are padded by speech_pad_ms each side (default 400)
            speech_pad_ms=400,
        ),
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


if __name__ == "__main__":

    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    port = os.getenv("PORT", 7860)
    
    logger.info("Launching Gradio UI")
    logger.info(f"Available at http://{server_name}:{port}")
    stream.ui.launch(
        server_port=port, 
        server_name=server_name,
        ssl_verify=False,
        debug=True
    )
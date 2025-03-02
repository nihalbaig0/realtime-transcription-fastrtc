import os
import logging

import gradio as gr
import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available
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

async def transcribe(stream, audio: tuple[int, np.ndarray]):
    sample_rate, audio_array = audio
    logger.info(f"Sample rate: {sample_rate}Hz, Shape: {audio_array.shape}")
    
    # Convert to mono if stereo
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    
    audio_array = audio_array.astype(np_dtype)
    audio_array /= np.max(np.abs(audio_array))
    
    if stream is not None:
        stream = np.concatenate((stream, audio_array))
    else:
        stream = audio_array

    outputs = transcribe_pipeline(
        {"sampling_rate": sample_rate, "raw": audio_array},
        chunk_length_s=10,
        batch_size=1,
        generate_kwargs={
            'task': 'transcribe',
            'language': 'english',
        },
        #return_timestamps="word"
    )
    return stream, outputs["text"].strip()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Audio Stream", streaming=True)
        with gr.Column():
            transcript = gr.Textbox(label="Transcript", value="")
        
        state = gr.State()
        audio_input.stream(
            transcribe, 
            inputs=[state, audio_input], 
            outputs=[state, transcript],
            stream_every=2
        )

        clear_button = gr.Button("Clear")
        clear_button.click(
            lambda: None, # clear the state
            outputs=[state]
        ).then(
            lambda: "", # clear the transcript
            outputs=[transcript]
        )

if __name__ == "__main__":
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    port = os.getenv("PORT", 7860)
    demo.launch(
        server_name=server_name,
        server_port=port,
        debug=True
    )

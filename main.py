import streamlit as st
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import soundfile as sf
import tempfile
import io
st.set_page_config(layout="wide")
st.header("Pick an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])
st.audio(uploaded_file, format="audio/wav")
st.write("<hr>", unsafe_allow_html=True)

# instantiate pipeline
#pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# JIT compile the forward call - slow, but we only do once
# used cached function thereafter - super fast!!


if uploaded_file:
    audio_data, sample_rate = sf.read(io.BytesIO(uploaded_file.read()))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        sf.write(temp_audio_path, audio_data, sample_rate)

        pipeline = FlaxWhisperPipline(
        "openai/whisper-tiny", dtype=jnp.bfloat16, batch_size=16)  # , device='cuda:0')

    # Transcribe and return timestamps
        outputs = pipeline(temp_audio_path, task="transcribe", return_timestamps=True)
        st.write(outputs)

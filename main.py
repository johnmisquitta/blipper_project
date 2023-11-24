import streamlit as st
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import soundfile as sf
import tempfile
import io
import librosa
st.set_page_config(layout="wide")
st.header("Pick an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])
#st.audio(uploaded_file, format="audio/wav")
st.write("<hr>", unsafe_allow_html=True)

# instantiate pipeline
#pipeline = FlaxWhisperPipline("openai/whisper-base")

# JIT compile the forward call - slow, but we only do once
# used cached function thereafter - super fast!!

def process_audio(file):
    audio_data, sample_rate = librosa.load(file)
    return audio_data, sample_rate

if uploaded_file is not None:
        st.audio(uploaded_file)
        audio_data, sample_rate = process_audio(uploaded_file)
        st.success("Audio processed successfully!")

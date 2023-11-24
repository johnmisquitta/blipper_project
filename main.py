import streamlit as st
import jax.numpy as jnp
import soundfile as sf
st.set_page_config(layout="wide")
st.header("Pick an Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])
st.audio(uploaded_file, format="audio/wav")
st.write("<hr>", unsafe_allow_html=True)


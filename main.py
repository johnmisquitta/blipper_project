# import streamlit as st
# from whisper_jax import FlaxWhisperPipline
# import jax.numpy as jnp
# import soundfile as sf
# import tempfile
# import io
# import librosa
# st.set_page_config(layout="wide")
# st.header("Pick an Audio File")
# uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])
# #st.audio(uploaded_file, format="audio/wav")
# st.write("<hr>", unsafe_allow_html=True)

# # instantiate pipeline
# #pipeline = FlaxWhisperPipline("openai/whisper-base")

# # JIT compile the forward call - slow, but we only do once
# # used cached function thereafter - super fast!!

# def process_audio(file):
#     audio_data, sample_rate = librosa.load(file)
#     return audio_data, sample_rate

# if uploaded_file is not None:
#         st.audio(uploaded_file)
#         audio_data, sample_rate = process_audio(uploaded_file)
#         st.success("Audio uploaded successfully!")
#         pipeline = FlaxWhisperPipline("openai/whisper-tiny", dtype=jnp.bfloat16, batch_size=16)  # , device='cuda:0')

#         # Transcribe and return timestamps
#         outputs = pipeline(audio_data, task="transcribe", return_timestamps=True)
#         st.write(outputs)
#         st.success("Audio processed successfully!")
#############################working############################
import streamlit as st
#from whisper_jax import FlaxWhisperPipline
#import jax.numpy as jnp
import soundfile as sf
import tempfile
import io
import librosa   
import whisperx
import gc 
import streamlit as st
import os
import tempfile
import time
import pandas as pd

# st.title("Word Level Transcription")


# audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"]) #upload file
# audio_button = st.button("Play Audio", key="audio_button")
# def process_audio(file):
#     audio_data, sample_rate = librosa.load(file)
#     return audio_data, sample_rate
# transcription_dict=[]

# # Display the full transcript
# st.header("Transcript:")
# if audio_file is not None:
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_file_path = os.path.join(temp_dir.name, audio_file.name)
#     with open(temp_file_path, 'wb') as temp_file:
#         temp_file.write(audio_file.read())
#     st.audio(audio_file)
#     #audio_data, sample_rate = process_audio(audio_file)
#     st.success("Audio processed successfully!")
#     # Process the file using its path
#     #st.audio(audio_file, format='audio/*')



#     device = 'cpu'

#     # Process the file
# # st.audio(audio_file, format='audio/*')
#     batch_size = 16 # reduce if low on GPU mem
#     #compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
#     compute_type = "int8"

#     # 1. Transcribe with original whisper (batched)
#     model = whisperx.load_model("tiny",device, compute_type=compute_type)

#     audio = whisperx.load_audio(temp_file)
    #result = model.transcribe(temp_file, batch_size=batch_size)
    #st.write(result["segments"]) # before alignment

    # # delete model if low on GPU resources
    # # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # # 2. Align whisper output
    # model_a, metadata = whisperx.load_align_model(language_code=result["language"],device=device)
    # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # st.write(result)

    # #print(result["segments"]) # after alignment

    # # delete model if low on GPU resources
    # # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # # 3. Assign speaker labels
    # diarize_model = whisperx.DiarizationPipeline(use_auth_token="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")#, device=device)

    # # add min/max number of speakers if known
    # diarize_segments = diarize_model(audio)
    # # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # result = whisperx.assign_word_speakers(diarize_segments, result)
    # st.write(result["segments"])
    # #print(diarize_segments)
    # #print(result["segments"]) # segments are now assigned speaker IDs
    # transcription_dict =result["segments"]
    # st.write(transcription_dict)

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_transcription_at_timestamp(timestamp, transcription_dict):
    for item in transcription_dict:
        if item['start'] <= timestamp <= item['end']:
            #print(item['speaker'])
            return item
    return None

def load_transcription_and_highlight(timestamp, transcription_dict):
    selected_transcription = get_transcription_at_timestamp(timestamp, transcription_dict)

    if selected_transcription:

        words_at_timestamp = [word_info['word'] for word_info in selected_transcription['words']
                               if word_info['start'] <= timestamp <= word_info['end']]
        highlighted_transcript = " ".join([f'<span style="background-color:  #aaffaa;">{word_info["word"]}</span>' if word_info["word"] in words_at_timestamp else (word_info["word"])
                                           for word_info in selected_transcription['words']])
        
        speaker=selected_transcription["speaker"]
        #print(speaker)
        #result_container.markdown(highlighted_transcript, unsafe_allow_html=True)
        #st.markdown(highlighted_transcript, unsafe_allow_html=True)
        return speaker,highlighted_transcript


    # else:
    #     st.warning("No transcription found for the given timestamp.")

def convert_speaker_name(speaker_name):
        my_list = ["#FFE15D","#FD841F", "blue", "green", "Purple"]

        if speaker_name.startswith("SPEAKER_"):
            try:
                index = int(speaker_name.split("_")[1])
                changed_speaker_name=f"Speaker {index + 1}"
                colour=my_list[index]
                return changed_speaker_name,colour
            except ValueError:
                return speaker_name,"yellow"
        else:
            return speaker_name,"yellow"
def main():
    st.title("Word Level Transcription")


    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"]) #upload file
    audio_button = st.button("Play Audio", key="audio_button")
    if audio_button:
        st.audio(audio_file, format="audio/mp3")
    transcription_dict=[]

    # Display the full transcript
    st.header("Transcript:")
    if audio_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, audio_file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(audio_file.read())


        

        # Process the file using its path
        #st.audio(temp_file_path, format='audio/*')

        device = 'cpu'

        # Process the file
    # st.audio(audio_file, format='audio/*')
        batch_size = 16 # reduce if low on GPU mem
        #compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        compute_type = "float32"

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("tiny",device, compute_type=compute_type)

        audio = whisperx.load_audio(temp_file_path)
        transcribe_result = model.transcribe(audio, batch_size=batch_size)
        st.write("transcribe")

        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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")#, device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)

        st.write(transcribe_result["segments"]) # before alignment
        result = whisperx.assign_word_speakers(diarize_segments, transcribe_result)
        st.write(result["segments"])

        df = pd.DataFrame(result["segments"])
        st.write(df)

        # Group the DataFrame by the "speaker" column
        grouped_df = df.groupby("speaker")


        # Access each group and print the result
        for speaker, group in grouped_df:
            st.write(f"Speaker: {speaker}")
            for index, row in group.iterrows():
                st.write(f"  Text: {row['text']}")

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    #     # 2. Align whisper output
    #     model_a, metadata = whisperx.load_align_model(language_code=transcribe_result["language"],device=device)
    #     result = whisperx.align(transcribe_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    #     st.write("allign")

    #     st.write(result)

    #     #print(result["segments"]) # after alignment

    #     # delete model if low on GPU resources
    #     # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    #     # 3. Assign speaker labels
    #     diarize_model = whisperx.DiarizationPipeline(use_auth_token="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")#, device=device)

    #     # add min/max number of speakers if known
    #     diarize_segments = diarize_model(audio)
    #     # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    #     st.write("result")


    #     result = whisperx.assign_word_speakers(diarize_segments, transcribe_result)
    #     st.write(result["segments"])
    #     #print(diarize_segments)
    #     #print(result["segments"]) # segments are now assigned speaker IDs
    #     transcription_dict =result["segments"]
    #     st.write("transcription dictionary")

    #     st.write(transcription_dict)







    # # Replace with the actual path to your audio file
    # # audio_file = "converted_audio.wav"
    

if __name__ == "__main__":
    main()

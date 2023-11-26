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
#import soundfile as sf
import tempfile
#import io
import librosa   
import whisperx
#import gc 
import streamlit as st
import os
import tempfile
#import time
import pandas as pd
import openai
#import pydub
import numpy as np
import torch
# st.title("Word Level Transcription")
st.set_page_config(layout="wide")


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



def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()




openai.api_key = "api key"

def main():
    st.header("Upload A File")



    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"]) #upload file
    #audio_button = st.button("Play Audio", key="audio_button")
    #if audio_button:
    #   st.audio(audio_file, format="audio/mp3")
    transcription_dict=[]

    # Display the full transcript
    st.header("Transcript")
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


        

        #audio = whisperx.load_audio(temp_file_path)
        audio, sample_rate = librosa.load(temp_file_path, sr=None, mono=True)

        # Optionally, you can resample the audio to a specific sample rate if needed
        # target_sample_rate = 16000
        # audio = librosa.resample(audio, sample_rate, target_sample_rate)
        # sample_rate = target_sample_rate

        # Convert NumPy array to torch.Tensor
        waveforms_tensor = torch.from_numpy(audio).float()

        # Convert torch.Tensor to NumPy array
        audio = waveforms_tensor.numpy()






        transcribe_result = model.transcribe(audio, batch_size=batch_size)
        st.write("language detected " + transcribe_result['language'])
        st.write("transcript")




        st.write(transcribe_result['segments'])


        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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")#, device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio)


        st.header("Diarization + Transcript")

        result = whisperx.assign_word_speakers(diarize_segments, transcribe_result)
        st.write(result["segments"])

        df = pd.DataFrame(result["segments"])
        st.write(df)

        # Group the DataFrame by the "speaker" column
        grouped_df = df.groupby("speaker")

        st.header("Split")
        # Access each group and print the result
        for speaker, group in grouped_df:

            st.write(f"Speaker: {speaker}")
            for index, row in group.iterrows():
                st.write(f"Text: {row['text']}")
                st.write(f"Text: {row}")


        st.header("Split + Merged")

        # for speaker, group in grouped_df:
        #     speaker_text = f"Speaker: {speaker}\n"
        #     for index, row in group.iterrows():
        #         speaker_text += f"{row['text']}\n"
        #     # Display the speaker text
        #     st.write(speaker_text)

    # For each speaker, create a column for their name and text
        grouped_df = df.groupby('speaker')

        # Get the unique list of speakers
        speakers = df['speaker'].unique()
        st.write(len(speakers))
        speaker_info = {}

        for i, speaker in enumerate(speakers, start=1):
            # Get the group corresponding to the current speaker
            group = grouped_df.get_group(speaker)

            # Create dynamic variable names (speaker1, speaker2, etc.)
            speaker_name = f"speaker{i}"
            text_name = f"text{i}"

            # Assign values to the dictionary
            speaker_info[speaker_name] = speaker
            speaker_info[text_name] = group['text'].tolist()

        # Print the variables
        #for i in range(1, len(speakers) + 1):
        col1, col2 = st.columns(2)



        with col1:
            if len(speakers)>=1:

                st.header(speaker_info[f'speaker{1}'])
                #st.write("This is content for column 2.")
                #st.write(f"Speaker {1}: {speaker_info[f'speaker{1}']}")
                st.write(f"Text {1}: {speaker_info[f'text{1}']}")
                st.header("GPT Prompt")
                quaries="perform these operations \n summary \n check mistakes \n query 3 \n query 4\n"
                text=speaker_info[f'text{1}']
                quaries_plus_text=quaries + ' '.join(text)

            
                prompt = st.text_area("Prompts:", quaries_plus_text)

                # Button to generate response
                if st.button("Submit 1"):
                    if prompt:
                        st.warning("add open ai Api to generate response.")

                        #recap = generate_response(prompt)
                        #st.markdown(f"**Recap:**\n{recap}")
                    else:
                        st.warning("Please enter a prompt.")
            else:
                st.warning("2nd speaker not found")


        with col2:
            if len(speakers)>=2:
                st.header(speaker_info[f'speaker{2}'])
                #st.write("This is content for column 2.")

                #st.write(f"Speaker {2}: {speaker_info[f'speaker{2}']}")
                st.write(f"Text {2}: {speaker_info[f'text{2}']}")
                st.header("GPT Prompt")


                quaries="perform these operations \n summary \n check mistakes \n query 3 \n query 4\n"
                text=speaker_info[f'text{2}']
                quaries_plus_text=quaries + ' '.join(text)


            
                prompt = st.text_area("Prompts:", quaries_plus_text)

                # Button to generate response
                if st.button("Submit 2"):
                    if prompt:
                        st.warning("add open ai Api to generate response.")

                        recap = generate_response(prompt)
                        #st.markdown(f"**Recap:**\n{recap}")
                    else:
                        st.warning("Please enter a prompt.")
            st.warning("2nd speaker not found")



        # # For each speaker, create a column for their name and text
        # for speaker in speakers:
        #     col_name, col_text = st.columns(2)  # 2 columns for each speaker

        #     col_name.write(f"{speaker}")

        #     group = grouped_df.get_group(speaker)

        #     speaker_text = "\n".join(group['text'])
        #     col_text.write(speaker_text)

        # col1, col2 = st.columns(2)

        # with col1:
        #     st.header("Column 1")
        #     st.write("This is content for column 1.")

        # with col2:
        #     st.header("Column 2")
        #     st.write("This is content for column 2.")







       

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    #     model_a, metadata = whisperx.load_align_model(language_code=transcribe_result["language"],device=device)
    #     result = whisperx.align(transcribe_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    #     st.write("allign")

    #     st.write(result)

    #     #print(result["segments"]) # after alignment

    #     # delete model if low on GPU resources
    #     # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

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







    # # audio_file = "converted_audio.wav"
    

if __name__ == "__main__":
    main()

import streamlit as st
import io
import soundfile as sf
from transformers import pipeline
import numpy as np

# Usa st.cache_resource (nuova API di caching) per caricare i modelli

@st.cache_resource
def load_asr_model():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    return asr

@st.cache_resource
def load_summary_model():
    summarizer = pipeline("text2text-generation", model="google/flan-t5-small")
    return summarizer

asr_model = load_asr_model()
summary_model = load_summary_model()

def summarize_meeting(text):
    prompt = (
        "Riassumi il seguente transcript della riunione ed estrai i principali action items come elenco puntato:\n\n"
        f"{text}\n\nRiassunto:"
    )
    result = summary_model(prompt, max_length=150, do_sample=False)
    return result[0]['generated_text']

def main():
    st.title("Riassunto Meeting & Estrazione Action Items")
    st.write("Carica un file audio della riunione o incolla il transcript per ottenere un riassunto ed i principali action items.")

    # Sezione per il caricamento del file audio
    uploaded_file = st.file_uploader("Carica file audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
    
    transcript = ""
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with st.spinner("Trascrivendo l'audio..."):
            audio_bytes = uploaded_file.read()
            # Leggi il file audio usando soundfile
            data, samplerate = sf.read(io.BytesIO(audio_bytes))
            # Se l'audio è stereo (o multi-canale), convertilo in mono
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = np.mean(data, axis=1)
            transcript = asr_model(data)["text"]
        st.subheader("Trascrizione")
        st.write(transcript)
    else:
        # Se non viene caricato un file, l'utente può inserire il testo manualmente
        transcript = st.text_area("O incolla qui il transcript della riunione", height=300)

    if st.button("Riassumi Riunione"):
        if transcript:
            with st.spinner("Elaborazione in corso..."):
                summary = summarize_meeting(transcript)
            st.subheader("Riassunto ed Action Items")
            st.write(summary)
        else:
            st.warning("Inserisci un transcript o carica un file audio.")

if __name__ == "__main__":
    main()

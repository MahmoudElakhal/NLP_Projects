import streamlit as st
import tempfile
import os

from whisper_model import transcribe
from predict import ToxicClassifier


st.set_page_config(
    page_title="Toxic Speech Detector",
    page_icon="🎙️",
    layout="wide"
)



st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .transcript-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-bottom: 12px;
    }

    .detected {
        font-weight: 700;
    }

    .safe {
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# LOAD MODEL

@st.cache_resource
def load_classifier():

    return ToxicClassifier()


classifier = load_classifier()


# HEADER


st.markdown(
    '<div class="main-title"> Toxic Speech Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Speech-to-text toxicity classification using Whisper + BiLSTM'
    '</div>',
    unsafe_allow_html=True
)


# SIDEBAR

with st.sidebar:

    st.header("System")

    st.success("Whisper: Loaded")
    st.success("BiLSTM: Loaded")
    st.success("Vocabulary: Loaded")

    st.divider()

    st.markdown("### Supported Categories")

    st.write("• Toxic")
    st.write("• Severe Toxic")
    st.write("• Obscene")
    st.write("• Threat")
    st.write("• Insult")
    st.write("• Identity Hate")

# MAIN INPUT


st.subheader("🎤 Record Speech")

audio_file = st.audio_input( "Click the microphone and record your speech")


if audio_file is not None:

    st.audio(audio_file,format="audio/wav")

    analyze_button = st.button("Analyze Speech",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        # SAVE AUDIO TEMPORARILY

        with tempfile.NamedTemporaryFile( delete=False,suffix=".wav") as temp_audio:

            temp_audio.write(audio_file.getvalue())

            audio_path = temp_audio.name


        try:

            # WHISPER
            with st.spinner("🎙️ Transcribing speech..."):

                transcript = transcribe(audio_path)

            # TRANSCRIPT
            st.subheader("📝 Transcript")

            if transcript:

                st.markdown(f""" <div class="transcript-box">{transcript} </div> """,unsafe_allow_html=True)

            else:

                st.warning("No speech was detected.")

                st.stop()


            # TOXICITY MODEL

            with st.spinner("Analyzing toxicity..."):

                results = classifier.predict(transcript)


            # RESULTS

            st.subheader("Toxicity Analysis")

            # Overall status

            any_toxic = any( result["prediction"]for result in results.values())


            if any_toxic:

                st.error("Potential toxic content detected")

            else:

                st.success("No toxic content detected")


            st.divider()

            # DISPLAY RESULTS
            cols = st.columns(3)

            for index, (label, result) in enumerate(results.items()):

                probability = (result["probability"])

                prediction = (result["prediction"])

                percentage = probability * 100

                col = cols[index % 3]

                with col:

                    st.markdown(f"### {label.replace('_', ' ').title()}")

                    st.progress(min(probability, 1.0))

                    st.metric("Probability",f"{percentage:.1f}%")

                    if prediction:
                        st.error("DETECTED")

                    else:

                        st.success(" Not detected")


        finally:

            # DELETE TEMP FILE


            if os.path.exists(audio_path):

                os.remove(audio_path)
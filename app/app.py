import streamlit as st
import numpy as np
import os
import librosa
import soundfile
import joblib
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import time

# Set page config
st.set_page_config(
    page_title="Audio Emotion Recognition",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'emotion_classification.joblib')
    return joblib.load(model_path)

# Cache the feature extraction function
@st.cache_data
def extract_advanced_features(file_name: str) -> np.ndarray:
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            # Ensure minimum duration
            target_length = 3 * sample_rate
            if len(X) < target_length:
                X = np.pad(X, (0, target_length - len(X)), mode='reflect')
            elif len(X) > target_length:
                X = X[:target_length]

            features = []

            # 1. Enhanced time-domain features
            features.extend([
                np.mean(np.abs(X)),
                np.std(X),
                np.max(np.abs(X)),
                np.sum(np.square(X)),
                np.mean(np.square(X)),
                scipy.stats.skew(X),
                scipy.stats.kurtosis(X),
                np.sum(np.diff(X) > 0),
                np.percentile(np.abs(X), 90)
            ])

            # 2. Enhanced spectral features
            stft = librosa.stft(X)
            stft_mag = np.abs(stft)
            stft_db = librosa.amplitude_to_db(stft_mag)

            features.extend([
                np.mean(stft_db),
                np.std(stft_db),
                scipy.stats.skew(stft_db.ravel()),
                scipy.stats.kurtosis(stft_db.ravel()),
                np.percentile(stft_db, 10),
                np.percentile(stft_db, 90)
            ])

            # 3. Enhanced MFCC features
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            for feat in [mfccs, mfcc_delta, mfcc_delta2]:
                features.extend([
                    np.mean(feat, axis=1),
                    np.std(feat, axis=1),
                    scipy.stats.skew(feat, axis=1),
                    scipy.stats.kurtosis(feat, axis=1)
                ])

            # 4. Enhanced rhythm features
            onset_env = librosa.onset.onset_strength(y=X, sr=sample_rate)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
            features.extend([
                tempo,
                len(beats) / (len(X) / sample_rate),
                np.std(np.diff(beats)) if len(beats) > 1 else 0
            ])

            # 5. Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(X)
            features.extend([
                np.mean(y_harmonic),
                np.std(y_harmonic),
                np.mean(y_percussive),
                np.std(y_percussive)
            ])

            # 6. Enhanced mel spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec)

            features.extend([
                np.mean(mel_spec_db, axis=1),
                np.std(mel_spec_db, axis=1),
                np.max(mel_spec_db, axis=1),
                np.min(mel_spec_db, axis=1)
            ])

            # 7. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=X, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=X, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate)[0]

            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])

            # Flatten and concatenate all features
            features = np.concatenate([np.array(f).flatten() for f in features])

            # Handle NaN and Inf values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def plot_emotion_probabilities(probabilities, emotions):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotions, y=probabilities)
    plt.title('Emotion Prediction Probabilities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# Main app
st.title('Audio Emotion Recognition')
st.divider()

# Load model
with st.spinner('Loading model...'):
    model = load_model()

# File upload
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])

# Process file as soon as it's uploaded
if uploaded_file:
    st.audio(uploaded_file)
    st.divider()
    
    # Save uploaded file
    with st.spinner('Processing audio file...'):
        # Create temporary file path
        temp_path = os.path.join(os.getcwd(), "temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Extracting audio features...")
        features = extract_advanced_features(temp_path)
        progress_bar.progress(50)
        
        if features is not None:
            status_text.text("Making prediction...")
            # Reshape features for prediction
            features = features.reshape(1, -1)
            
            try:
                # Get prediction and probabilities
                prediction = model.predict(features)
                probabilities = model.predict_proba(features)[0]
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Clean up temporary file
                os.remove(temp_path)
                
                # Display results
                st.divider()
                st.subheader("Results")
                
                # Emotion mapping
                emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
                predicted_emotion = emotions[prediction[0]]
                confidence = np.max(probabilities) * 100
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Emotion", predicted_emotion)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Plot probability distribution
                st.divider()
                st.subheader("Emotion Probability Distribution")
                fig = plot_emotion_probabilities(probabilities, emotions)
                st.pyplot(fig)
            except ValueError as e:
                st.error("Error: The extracted features are not compatible with the model. Please ensure you're using the correct model version.")
                st.error(str(e))
        else:
            st.error("Failed to extract features from the audio file.")

# Add information about the model
st.divider()
st.subheader("About the Model")
st.write("""
This emotion recognition system uses advanced audio feature extraction and machine learning
to classify emotions in speech. It analyzes various aspects of the audio signal including:
- Temporal features
- Spectral characteristics
- Mel-frequency cepstral coefficients (MFCCs)
- Rhythm and tempo
- Harmonic and percussive components

The model is trained on a dataset of emotional speech recordings and can classify
eight different emotional states: neutral, calm, happy, sad, angry, fearful, disgust,
and surprised.
""")
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import joblib
import librosa
import numpy as np
import os
import gdown

# ==========================================
# AUTOMATIC GOOGLE DRIVE FOLDER DOWNLOADER
# ==========================================
@st.cache_resource
def download_models_from_drive_folder():
    target_dir = "models"
    required_files = [
        os.path.join(target_dir, "model_1_resnet18.pth"),
        os.path.join(target_dir, "model_2_lightgbm.pkl"),
        os.path.join(target_dir, "meta_learner.pkl")
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        with st.spinner("📥 Downloading models from Google Drive folder... (This takes 1-2 minutes on first boot)"):
            folder_id = '1f-_kq2aHau52gtolJWLaScXhBNXcZZ5a'
            try:
                gdown.download_folder(id=folder_id, output=target_dir, quiet=False)
                st.success("✅ Models successfully synchronized from cloud storage!")
            except Exception as e:
                st.error(f"⚠️ Automatic download failed. Error: {e}")

download_models_from_drive_folder()

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Deepfake Audio Detection Engine")
st.markdown("Upload a `.wav` or `.mp3` file to analyze if the audio is Genuine or AI-Generated.")

# ==========================================
# 2. MODEL BLUEPRINTS
# ==========================================
class RobustAudioResNet(nn.Module):
    def __init__(self):
        super(RobustAudioResNet, self).__init__()
        self.resnet = models.resnet18(weights=None) 
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        
    def forward(self, x):
        return self.resnet(x)

# ==========================================
# 3. MODEL CACHING
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cpu')
    resnet = RobustAudioResNet()
    resnet.load_state_dict(torch.load("models/model_1_resnet18.pth", map_location=device))
    resnet.eval()
    lgbm = joblib.load("models/model_2_lightgbm.pkl")
    meta_learner = joblib.load("models/meta_learner.pkl")
    return resnet, lgbm, meta_learner

try:
    resnet_model, lgb_model, meta_model = load_models()
    st.success("✅ Models loaded successfully into memory.")
except Exception as e:
    st.error(f"⚠️ Error loading models. Details: {e}")

# ==========================================
# 4. AUDIO FEATURE EXTRACTION (RESTORED TO TEST SCRIPT LOGIC)
# ==========================================
def process_audio(file_buffer):
    TARGET_SR = 16000
    DURATION = 5.0
    TOTAL_SAMPLES = int(TARGET_SR * DURATION)
    
    # Load audio
    audio, sr = librosa.load(file_buffer, sr=TARGET_SR, mono=True)
    
    # Exact padding logic from test_inference.py
    if len(audio) < TOTAL_SAMPLES:
        audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), mode='constant')
    else:
        audio = audio[:TOTAL_SAMPLES]
        
    # Spatial Branch
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    spatial_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spatial_tensor = F.interpolate(spatial_tensor, size=(128, 157), mode='bilinear', align_corners=False)
    
    # Tabular Branch (Exact features from test_inference.py)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=20), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=TARGET_SR, n_chroma=12), axis=1)
    mel_12 = np.mean(librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=12), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR), axis=1)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=TARGET_SR), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR, roll_percent=0.85), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio), axis=1)
    
    tabular_features = np.concatenate([mfccs, chroma, mel_12, centroid, bandwidth, rolloff, zcr]).reshape(1, -1)
    
    return spatial_tensor, tabular_features

# ==========================================
# 5. UI INFERENCE PIPELINE
# ==========================================
uploaded_file = st.file_uploader("Upload an audio sample...", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔍 Analyze Audio"):
        with st.spinner("Extracting 48 acoustic features and running ensemble..."):
            try:
                tensor_spec, tabular_features = process_audio(uploaded_file)
                
                with torch.no_grad():
                    resnet_out = resnet_model(tensor_spec)
                    p_deepfake_resnet = F.softmax(resnet_out, dim=1)[:, 1].item()
                    
                p_deepfake_lgb = lgb_model.predict_proba(tabular_features)[:, 1][0]
                
                meta_input = np.array([[p_deepfake_resnet, p_deepfake_lgb]])
                p_deepfake_final = meta_model.predict_proba(meta_input)[:, 1][0]
                
                # Default optimal boundary
                THRESHOLD = 0.5000 
                
                st.markdown("---")
                st.subheader("Analysis Results")
                
                if p_deepfake_final >= THRESHOLD:
                    st.error("🚨 **DEEPFAKE DETECTED**")
                    st.write(f"**Confidence Score:** {p_deepfake_final:.2%}")
                else:
                    st.success("✅ **GENUINE HUMAN SPEECH**")
                    st.write(f"**Confidence Score:** {(1.0 - p_deepfake_final):.2%}")
                    
                with st.expander("See individual model scores (Probability of Deepfake)"):
                    st.write(f"- **ResNet18 Score:** {p_deepfake_resnet:.2%}")
                    st.write(f"- **LightGBM Score:** {p_deepfake_lgb:.2%}")
                    st.write(f"- **Meta-Ensemble Score:** {p_deepfake_final:.2%}")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

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
                st.info("Please verify that your Google Drive folder access is set to 'Anyone with the link'.")

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
    st.error(f"⚠️ Error loading models. Please check your 'models/' folder. Details: {e}")

# ==========================================
# 4. AUDIO FEATURE EXTRACTION
# ==========================================
# ==========================================
# 4. AUDIO FEATURE EXTRACTION (UPDATED)
# ==========================================
def process_audio(file_buffer):
    # Load the audio at native length
    y, sr = librosa.load(file_buffer, sr=16000)
    target_length = sr * 5  # 80,000 samples for 5 seconds
    
    # THE FIX: Loop the audio instead of padding with zeros
    if len(y) < target_length:
        # Repeat the audio array enough times to exceed 5 seconds
        y = np.tile(y, int(np.ceil(target_length / len(y))))
        
    # Enforce exact 5-second truncation
    y = y[:target_length]
        
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    tensor_spec = torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0).float()
    tensor_spec = F.interpolate(tensor_spec, size=(128, 157), mode='bilinear', align_corners=False)
    
    features = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfccs, axis=1)) 
    
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) 
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))) 
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) 
    features.append(np.mean(librosa.feature.zero_crossing_rate(y))) 
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1)) 
    
    mel_means = np.mean(mel_spec_db, axis=1)
    features.extend(np.resize(mel_means, 12)) 
    
    tabular_features = np.array(features[:48]).reshape(1, -1)
    
    return tensor_spec, tabular_features

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
                
                # REVERTED: Index 1 officially maps to Deepfake
                with torch.no_grad():
                    resnet_out = resnet_model(tensor_spec)
                    p_deepfake_resnet = F.softmax(resnet_out, dim=1)[:, 1].item()
                    
                p_deepfake_lgb = lgb_model.predict_proba(tabular_features)[:, 1][0]
                
                meta_input = np.array([[p_deepfake_resnet, p_deepfake_lgb]])
                p_deepfake_final = meta_model.predict_proba(meta_input)[:, 1][0]
                
                THRESHOLD = 0.5000
                
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # REVERTED: Correct logic restored. High probability = Deepfake.
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

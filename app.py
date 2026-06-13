import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import joblib
import librosa
import numpy as np

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
# 3. LEGACY CACHING (The Fix)
# ==========================================
# Replaced @st.cache_resource with the legacy PyTorch model cache
@st.cache(allow_output_mutation=True)
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
    st.success("✅ Models loaded successfully from disk.")
except Exception as e:
    st.error(f"⚠️ Error loading models. Please check your 'models/' folder. Details: {e}")

# ==========================================
# 4. AUDIO FEATURE EXTRACTION
# ==========================================
def process_audio(file_buffer):
    y, sr = librosa.load(file_buffer, sr=16000, duration=5.0)
    
    if len(y) < sr * 5:
        y = np.pad(y, (0, sr * 5 - len(y)))
        
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
    
    # Removed modern button styling for legacy support
    if st.button("🔍 Analyze Audio"):
        with st.spinner("Extracting 48 acoustic features and running ensemble..."):
            try:
                tensor_spec, tabular_features = process_audio(uploaded_file)
                
                with torch.no_grad():
                    resnet_out = resnet_model(tensor_spec)
                    m1_prob = F.softmax(resnet_out, dim=1)[:, 1].item()
                    
                m2_prob = lgb_model.predict_proba(tabular_features)[:, 1][0]
                
                meta_input = np.array([[m1_prob, m2_prob]])
                final_prob = meta_model.predict_proba(meta_input)[:, 1][0]
                
                THRESHOLD = 0.5000 
                
                # Replaced st.divider() with legacy markdown line
                st.markdown("---")
                st.subheader("Analysis Results")
                
                if final_prob >= THRESHOLD:
                    st.error("🚨 **DEEPFAKE DETECTED**")
                    st.write(f"**Confidence Score:** {final_prob:.2%}")
                else:
                    st.success("✅ **GENUINE AUDIO**")
                    st.write(f"**Deepfake Probability:** {final_prob:.2%}")
                    
                with st.expander("See individual model scores"):
                    st.write(f"- **ResNet18 Score:** {m1_prob:.2%}")
                    st.write(f"- **LightGBM Score:** {m2_prob:.2%}")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
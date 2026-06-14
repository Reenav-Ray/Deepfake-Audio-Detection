import os
import sys
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. CONFIGURATION & OPTIMIZED THRESHOLD
# ==========================================
MODELS_DIR = "models"
TARGET_SR = 16000
DURATION = 5.0
TOTAL_SAMPLES = int(TARGET_SR * DURATION)  # 80,000 samples

# Set this to the exact Youden's J threshold calculated during training
OPTIMAL_THRESHOLD = 0.50  # Adjust if your training script output a custom threshold


# ==========================================
# 2. RESNET18 ARCHITECTURE DEFINITION
# ==========================================
class RobustAudioResNet(nn.Module):
    def __init__(self):
        super(RobustAudioResNet, self).__init__()
        # Load a standard ResNet18 backbone without pretrained weights
        self.resnet = models.resnet18(weights=None)
        
        # Re-engineer the input layer: Collapse 3 RGB channels into 1 Spectrogram channel
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Re-engineer output layer: Map to 2 units (Genuine vs Deepfake)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)


# ==========================================
# 3. CORE FEATURE EXTRACTION PIPELINE
# ==========================================
def preprocess_and_extract(file_path):
    """
    Ingests, downsamples, normalizes, and splits an audio file into 
    spatial spectrogram and tabular features.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
        
    # Ingest and force monophonic downsampling to 16kHz
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # Temporal Standardization (Strictly enforce 5.0 seconds)
    if len(audio) < TOTAL_SAMPLES:
        # Pad with silence if too short
        audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), mode='constant')
    else:
        # Truncate if too long
        audio = audio[:TOTAL_SAMPLES]
        
    # --- Branch A: Spatial Feature Extraction (Log-dB Spectrogram) ---
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transform to PyTorch Tensor shape [Batch=1, Channel=1, Height=128, Width=Time]
    spatial_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Bilinear interpolate to exact dimensions expected by ResNet (128 x 157)
    spatial_tensor = F.interpolate(spatial_tensor, size=(128, 157), mode='bilinear', align_corners=False)
    
    # --- Branch B: Tabular Feature Extraction (48 Key Metrics) ---
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=20), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=TARGET_SR, n_chroma=12), axis=1)
    mel_12 = np.mean(librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=12), axis=1)
    
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR), axis=1)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=TARGET_SR), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR, roll_percent=0.85), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio), axis=1)
    
    # Concatenate into the 48-dimensional vector
    tabular_vector = np.concatenate([mfccs, chroma, mel_12, centroid, bandwidth, rolloff, zcr])
    tabular_vector = tabular_vector.reshape(1, -1)  # Shape for single sample inference
    
    return spatial_tensor, tabular_vector


# ==========================================
# 4. MODEL LOADING & INFERENCE EXECUTION
# ==========================================
def run_predict(audio_path):
    print(f"\n[Processing] Analyzing sample: {os.path.basename(audio_path)}")
    
    # 1. Extract the features
    try:
        spatial_in, tabular_in = preprocess_and_extract(audio_path)
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    # 2. Paths to files
    resnet_path = os.path.join(MODELS_DIR, "model_1_resnet18.pth")
    lgb_path = os.path.join(MODELS_DIR, "model_2_lightgbm.pkl")
    meta_path = os.path.join(MODELS_DIR, "meta_learner.pkl")

    # 3. Load weights safely
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet_model = RobustAudioResNet()
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet_model.to(device)
    resnet_model.eval()
    
    lgb_model = joblib.load(lgb_path)
    meta_learner = joblib.load(meta_path)

    # 4. Generate Base Model Probabilities
    with torch.no_grad():
        spatial_in = spatial_in.to(device)
        resnet_logits = resnet_model(spatial_in)
        # Apply softmax to get standard probability distributed across classes
        resnet_probs = F.softmax(resnet_logits, dim=1).cpu().numpy()
        p_resnet = resnet_probs[0, 1]  # Index 1 corresponds to Deepfake

    # LightGBM outputs probability array [prob_class_0, prob_class_1]
    lgb_probs = lgb_model.predict_proba(tabular_in)
    p_lgb = lgb_probs[0, 1]

    # 5. Execute Stacked Meta-Learner Layer
    meta_input = np.array([[p_resnet, p_lgb]])
    final_deepfake_prob = meta_learner.predict_proba(meta_input)[0, 1]

    # 6. Apply Decision Boundary
    verdict = "DEEPFAKE / SYNTHETIC" if final_deepfake_prob >= OPTIMAL_THRESHOLD else "GENUINE HUMAN SPEECH"
    confidence = final_deepfake_prob if final_deepfake_prob >= OPTIMAL_THRESHOLD else (1.0 - final_deepfake_prob)

    # --- OUTPUT RESULTS DASHBOARD ---
    print("=" * 60)
    print("             DEEPFAKE DETECTION ENGINE RESULTS              ")
    print("=" * 60)
    print(f"   Final Decision     :  {verdict}")
    print(f"   Confidence Score   :  {confidence * 100:.2f}%")
    print("-" * 60)
    print(f"   ResNet18 Prob      :  {p_resnet * 100:.2f}% (Spatial Branch)")
    print(f"   LightGBM Prob      :  {p_lgb * 100:.2f}% (Tabular Branch)")
    print(f"   Ensemble Meta Prob :  {final_deepfake_prob * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # Allows execution via command line arguments: python test_inference.py path_to_file.wav
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        run_predict(target_file)
    else:
        print("💡 Usage: python test_inference.py <path_to_audio_file>")
        print("⚠️ No file specified. Looking for a default fallback file 'test_sample.wav'...")
        run_predict("test_sample.wav")

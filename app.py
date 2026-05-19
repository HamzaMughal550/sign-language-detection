import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

st.set_page_config(page_title="PSL Recognition", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM KERAS LAYER & WORKAROUNDS ---
# Patch Keras Dense layer to ignore 'quantization_config' which causes issues in Keras 3
_original_dense_init = tf.keras.layers.Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = _patched_dense_init

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads=4, ff_dim=256, rate=0.3, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, dim//heads))
        self.ffn = Sequential([Dense(ff_dim, activation='relu'), Dense(dim)])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.dp1 = Dropout(rate)
        self.dp2 = Dropout(rate)
        
    def call(self, x, training=False):
        a = self.dp1(self.att(x, x, training=training), training=training)
        o = self.ln1(x + a)
        f = self.dp2(self.ffn(o), training=training)
        return self.ln2(o + f)
        
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'dim': self.dim, 'heads': self.heads, 'ff_dim': self.ff_dim, 'rate': self.rate})
        return cfg

# --- CACHE RESOURCES ---
@st.cache_resource
def load_all_models():
    model = tf.keras.models.load_model(
        'multistream_best.keras', 
        custom_objects={'TransformerBlock': TransformerBlock}
    )
    classes = np.load('classes.npy', allow_pickle=True)
    
    base_options_face = python.BaseOptions(model_asset_path='face.task')
    options_face = vision.FaceLandmarkerOptions(base_options=base_options_face, num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(options_face)
    
    base_options_pose = python.BaseOptions(model_asset_path='pose.task')
    options_pose = vision.PoseLandmarkerOptions(base_options=base_options_pose, num_poses=1)
    pose_detector = vision.PoseLandmarker.create_from_options(options_pose)
    
    base_options_hand = python.BaseOptions(model_asset_path='hand.task')
    options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(options_hand)
    
    return model, classes, face_detector, pose_detector, hand_detector

try:
    model, classes, face_detector, pose_detector, hand_detector = load_all_models()
except Exception as e:
    st.error(f"Error loading models. Please ensure .task, .keras, and .npy files are in the directory.\n\nDetails: {e}")
    st.stop()

# --- STATE MANAGEMENT ---
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'sequence_data' not in st.session_state:
    st.session_state.sequence_data = None
if 'frames_recorded' not in st.session_state:
    st.session_state.frames_recorded = 0
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False
if 'cam' not in st.session_state:
    st.session_state.cam = None

def toggle_recording():
    if not st.session_state.is_recording:
        st.session_state.is_recording = True
        st.session_state.sequence_data = []
        st.session_state.frames_recorded = 0
        st.session_state.run_webcam = True
    else:
        st.session_state.is_recording = False

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
<style>
    /* Dark Theme & Typography */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide the white top header and move content up */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Styled Containers */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(25, 28, 36, 0.6);
        border-radius: 12px;
        padding: 10px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161A22;
        border-right: 1px solid #2D333B;
    }
    
    /* Force sidebar text to be visible */
    section[data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1f232b !important;
        color: #ffffff !important;
        border: 1px solid #3b4252 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 255, 127, 0.2) !important;
        border-color: #00FF7F !important;
        color: #00FF7F !important;
        background-color: #2a2f3a !important;
    }
    
    /* Prediction Box styling */
    .prediction-box {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #4a4e69;
    }
    .prediction-text {
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- APP LAYOUT ---
st.title("Pakistani Sign Language (PSL) Translator")
st.markdown("Live dual-feed AI translator. Real-time sign recognition with intelligent frame filtering.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    run_webcam_cb = st.checkbox("Turn Webcam On/Off", key="run_webcam", value=True)
    st.markdown("---")
    st.markdown("**How to use:**\n1. Turn on Webcam\n2. Click 'Start Recording'\n3. Perform your sign\n4. Click 'Stop Recording'\n5. Click 'Predict Sign'")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📷 CAMERA FEED")
    camera_placeholder = st.empty()
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        btn_text = "🔴 Stop Recording" if st.session_state.is_recording else "🟢 Start Recording"
        st.button(btn_text, on_click=toggle_recording, use_container_width=True)
    with btn_col2:
        predict_btn = st.button("✨ Predict Sign", type="primary", use_container_width=True)

with col2:
    st.markdown("### 🎯 DETECTED SENTENCE")
    pred_placeholder = st.empty()
    pred_placeholder.markdown("<div class='prediction-box'><p class='prediction-text' style='font-size:18px;'>Ready. Press Start Recording.</p></div>", unsafe_allow_html=True)
    
    st.markdown("### 🦴 SKELETON PREVIEW")
    skeleton_placeholder = st.empty()


# --- PREDICTION LOGIC ---
if predict_btn:
    if st.session_state.sequence_data is not None and len(st.session_state.sequence_data) > 0:
        pred_placeholder.markdown("<div class='prediction-box'><p class='prediction-text' style='font-size:20px;'>Analyzing Frames...</p></div>", unsafe_allow_html=True)
        
        seq = np.array(st.session_state.sequence_data).copy()
        
        # Smart Frame Filtering (Longest Active Segment)
        hands_present = np.any(seq[:, 1533:1659] != 0, axis=1)
        padded = np.pad(hands_present, (1, 1), mode='constant', constant_values=False)
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) > 0:
            lengths = ends - starts
            best_idx = np.argmax(lengths)
            start_idx = max(0, starts[best_idx] - 5) # 5 frame buffer
            end_idx = min(len(seq), ends[best_idx] + 5)
            seq = seq[start_idx:end_idx]
            
        # Normalization (SPOTER-style)
        if len(seq) > 0:
            T = seq.shape[0]
            root = seq[:, 0:3]
            for i in range(0, seq.shape[1], 3):
                seq[:, i:i+3] -= root
                
            scale = np.linalg.norm(seq.reshape(T, -1), axis=1, keepdims=True)
            scale[scale < 1e-6] = 1.0
            seq = seq / scale
            
            # Pad sequence to 200 frames (Model Input Shape)
            TARGET_FRAMES = 200
            if seq.shape[0] < TARGET_FRAMES:
                pad_length = TARGET_FRAMES - seq.shape[0]
                seq = np.pad(seq, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)
            elif seq.shape[0] > TARGET_FRAMES:
                seq = seq[:TARGET_FRAMES, :]
            
            # Stream Split
            face_s = seq[np.newaxis, :, :1434]
            pose_s = seq[np.newaxis, :, 1434:1533]
            hand_s = np.concatenate([seq[np.newaxis, :, 1533:1596],
                                     seq[np.newaxis, :, 1596:1659]], axis=-1)
                                     
            # Predict
            probs = model.predict([face_s, pose_s, hand_s], verbose=0)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_classes = [classes[i] for i in top3_idx]
            top3_probs = probs[top3_idx]
            
            # Display Results
            if top3_probs[0] > 0.70:
                pred_placeholder.markdown(f"<div class='prediction-box'><p class='prediction-text'>{top3_classes[0]}</p><p style='color:#a8ff78; margin-top:5px;'>Confidence: {top3_probs[0]*100:.1f}%</p></div>", unsafe_allow_html=True)
            else:
                pred_placeholder.markdown(f"<div class='prediction-box'><p class='prediction-text' style='color:#ffcccb;'>{top3_classes[0]}</p><p style='color:#ffcccb; margin-top:5px;'>Low Confidence: {top3_probs[0]*100:.1f}%</p></div>", unsafe_allow_html=True)
        else:
            pred_placeholder.error("No active sign detected in the recording.")
    else:
        pred_placeholder.error("No recording found. Please record a sign first.")


# --- WEBCAM & RECORDING LOGIC ---
if st.session_state.run_webcam:
    if st.session_state.cam is None:
        st.session_state.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    if not st.session_state.cam.isOpened():
        st.error("Camera could not be opened. Please check permissions.")
    else:
        while st.session_state.run_webcam:
            ret, frame = st.session_state.cam.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            
            # Enhance Brightness slightly for normal video
            camera_frame = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=15)
            camera_frame = cv2.flip(camera_frame, 1)
            
            # MediaPipe tasks processing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            face_result = face_detector.detect(mp_image)
            pose_result = pose_detector.detect(mp_image)
            hand_result = hand_detector.detect(mp_image)
            
            # Create a completely black frame for skeleton
            skeleton_frame = np.zeros((h, w, c), dtype=np.uint8)
            features = np.zeros(1659)
            
            # Extract and Draw Face
            if face_result.face_landmarks:
                for i, lm in enumerate(face_result.face_landmarks[0]):
                    features[i*3:i*3+3] = [lm.x, lm.y, lm.z]
                    cv2.circle(skeleton_frame, (int(lm.x * w), int(lm.y * h)), 1, (100, 100, 100), -1)
                    
            # Extract and Draw Pose
            if pose_result.pose_landmarks:
                for i, lm in enumerate(pose_result.pose_landmarks[0]):
                    features[1434+i*3:1434+i*3+3] = [lm.x, lm.y, lm.z]
                    cv2.circle(skeleton_frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                    
            # Extract and Draw Hand
            if hand_result.hand_landmarks:
                for idx, handedness in enumerate(hand_result.handedness):
                    is_left = handedness[0].category_name == 'Left'
                    offset = 1533 if is_left else 1596
                    color = (255, 0, 0) if is_left else (0, 0, 255) # Red/Blue for different hands
                    for i, lm in enumerate(hand_result.hand_landmarks[idx]):
                        features[offset+i*3:offset+i*3+3] = [lm.x, lm.y, lm.z]
                        cv2.circle(skeleton_frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)
                        
            # Mirror skeleton frame
            skeleton_frame = cv2.flip(skeleton_frame, 1)
            
            # Recording Logic
            if st.session_state.is_recording:
                st.session_state.sequence_data.append(features)
                st.session_state.frames_recorded += 1
                
                # Draw recording indicator on Camera Feed
                cv2.circle(camera_frame, (40, 40), 10, (255, 0, 0), -1) # Red recording dot
                cv2.putText(camera_frame, f"REC ({st.session_state.frames_recorded})", (60, 48), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            # Render both feeds simultaneously
            camera_placeholder.image(camera_frame)
            skeleton_placeholder.image(skeleton_frame)

else:
    if st.session_state.cam is not None:
        st.session_state.cam.release()
        st.session_state.cam = None

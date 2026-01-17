import streamlit as st
import whisper
import os
import tempfile
import time
import analysis_engine
import video_processor
import ai_engine

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PulsePoint AI | 34min -> 3 Reels",
    page_icon="⚡",
    layout="wide"
)

# --- CUSTOM CSS (VISIBILITY FIX) ---
st.markdown("""
    <style>
    /* 1. FORCE LIGHT THEME BACKGROUND */
    .stApp {
        background: linear-gradient(to bottom, #F0F8FF, #FFFFFF);
    }
    
    /* 2. FORCE MAIN TEXT TO BE DARK */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    p, div, span, label {
        color: #2C3E50 !important; /* Dark Grey for readability on white */
    }
    
    /* 3. CRITICAL FIX: FORCE UPLOAD BOX TEXT TO BE WHITE */
    /* This overrides the rule above specifically for the dark upload box */
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p {
        color: #FFFFFF !important;
    }
    
    /* 4. SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E1E4E8;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #2C3E50 !important;
    }
    
    /* 5. BUTTONS */
    .stButton>button {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 180, 219, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 180, 219, 0.3);
    }
    
    /* 6. METRIC CARDS */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #D1F2EB;
        border-left: 5px solid #2ECC71;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label { color: #555 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #000 !important; }
    
    /* 7. VIDEO CONTAINERS */
    .stVideo {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* 8. PROGRESS BAR */
    .stProgress > div > div > div > div {
        background-color: #2ECC71;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10609/10609040.png", width=80)
    st.title("PulsePoint AI")
    st.caption("🚀 Batch Process Long Videos")
    
    st.markdown("---")
    api_key = st.text_input("🔑 Gemini API Key", type="password")
    
    st.markdown("---")
    model_size = st.selectbox("🧠 Whisper Model", ["base", "small"], index=0)
    st.info("💡 **Tip:** Use 'Small' for better accuracy on long videos.")

# --- LOAD WHISPER ---
@st.cache_resource
def load_whisper(size):
    return whisper.load_model(size)

if model_size:
    with st.spinner(f"💧 Loading AI Models ({model_size})..."):
        model = load_whisper(model_size)

# --- MAIN UI ---
st.title("⚡ PulsePoint AI: Batch Processor")
st.markdown("##### 🎨 Transform your **34-minute video** into **Viral Shorts** with one click.")

# Styled File Uploader
uploaded_file = st.file_uploader("📂 Upload Long MP4", type=["mp4", "mov"])

if uploaded_file and api_key:
    # Use full width for results
    st.subheader("📺 Original Video")
    st.video(uploaded_file)
    
    # Big Call to Action
    st.markdown("---")
    if st.button("🚀 IGNITE ENGINE (Generate 3 Reels)", type="primary"):
        
        # Styled Status Container
        status = st.status("⚙️ **PulsePoint Engine Running...**", expanded=True)
        
        try:
            # 1. SAVE FILE
            status.write("💾 **Step 1:** Saving large video file...")
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name

            # 2. TRANSCRIBE
            status.write("📝 **Step 2:** Transcribing full 34 mins (Whisper)...")
            result = model.transcribe(video_path)
            transcript_text = result["text"]
            st.session_state['transcript'] = transcript_text

            # 3. INTELLIGENCE
            status.write("🧠 **Step 3:** Identifying Top 3 Viral Hooks (Gemini)...")
            segments = ai_engine.get_viral_segments(api_key, transcript_text)
            
            # 4. LOOP & GENERATE
            st.session_state['generated_reels'] = []
            
            progress_bar = status.progress(0)
            
            for i, seg in enumerate(segments):
                status.write(f"🎬 **Rendering Reel #{i+1}:** '{seg['headline']}'...")
                
                output_name = f"reel_{i+1}_{int(time.time())}.mp4"
                
                final_path = video_processor.smart_crop_to_vertical(
                    video_path,
                    float(seg['start_time']),
                    float(seg['end_time']),
                    seg['headline'],
                    output_name
                )
                
                if final_path:
                    st.session_state['generated_reels'].append({
                        "path": final_path,
                        "data": seg
                    })
                
                # Update progress (33%, 66%, 100%)
                progress_bar.progress((i + 1) / len(segments))

            status.update(label="✅ **ALL REELS GENERATED SUCCESSFULLY!**", state="complete", expanded=False)
            
            # Cleanup
            if os.path.exists(video_path):
                os.remove(video_path)

        except Exception as e:
            status.update(label="❌ **System Failure**", state="error")
            st.error(f"Error: {e}")

    # --- DISPLAY RESULTS GRID ---
    if 'generated_reels' in st.session_state:
        st.markdown("---")
        st.subheader("🔥 Your Viral Collection")
        
        # Create columns dynamically
        cols = st.columns(len(st.session_state['generated_reels']))
        
        for idx, reel in enumerate(st.session_state['generated_reels']):
            with cols[idx]:
                st.markdown(f"#### Reel #{idx+1}")
                st.video(reel['path'])
                
                # Metadata Card
                st.success(f"**Hook:** {reel['data']['headline']}")
                st.caption(f"💡 *{reel['data']['reason']}*")
                
                # Download Button
                with open(reel['path'], "rb") as file:
                    st.download_button(
                        label=f"⬇️ Download Reel #{idx+1}",
                        data=file,
                        file_name=f"viral_reel_{idx+1}.mp4",
                        mime="video/mp4"
                    )

elif not api_key:
    st.warning("👈 **Please enter your Gemini API Key in the sidebar and click enter to start.**")
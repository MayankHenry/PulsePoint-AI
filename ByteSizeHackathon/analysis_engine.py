import librosa
import numpy as np
import moviepy.editor as mp
import os

def extract_audio_from_video(video_path, audio_output="temp_audio.wav"):
    """
    Extracts audio from video to a WAV file for processing.
    """
    try:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_output, logger=None)
        return audio_output
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def detect_emotional_peaks(video_path, window_duration=60):
    """
    Analyzes audio energy to find the most 'viral' segment.
    Returns: (start_time, end_time) of the highest energy window.
    """
    print(f"🎧 Analyzing Audio Energy for: {os.path.basename(video_path)}...")
    
    # 1. Extract Audio
    audio_file = extract_audio_from_video(video_path)
    if not audio_file:
        return 0, window_duration

    # 2. Load Audio into Librosa
    # y = audio time series, sr = sample rate
    y, sr = librosa.load(audio_file, sr=None)
    
    # 3. Calculate RMS Energy (Loudness/Intensity)
    # hop_length defines how many samples per frame
    hop_length = 512
    rmse = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    # 4. Create a Timeline of Frames
    frames = range(len(rmse))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    # 5. Sliding Window Algorithm to find the "Loudest" 60 seconds
    # Convert window duration (seconds) to number of frames
    frames_per_sec = len(frames) / t[-1]
    window_size = int(window_duration * frames_per_sec)
    
    if window_size > len(rmse):
        window_size = len(rmse) # Handle short videos

    # Convolve allows us to sum up energy in the sliding window efficiently
    window_energy = np.convolve(rmse, np.ones(window_size), mode='valid')
    
    # Find the index of the maximum energy window
    max_energy_index = np.argmax(window_energy)
    
    # Convert index back to timestamps
    start_time = t[max_energy_index]
    end_time = start_time + window_duration
    
    # Cleanup temp file
    if os.path.exists(audio_file):
        os.remove(audio_file)
        
    print(f"🔥 Viral Segment Detected: {start_time:.2f}s to {end_time:.2f}s")
    return start_time, end_time

if __name__ == "__main__":
    # Test with your input video
    video = "input_video.mp4" 
    if os.path.exists(video):
        start, end = detect_emotional_peaks(video)
        print(f"✅ Suggestion: Crop from {start} to {end}")
    else:
        print("❌ input_video.mp4 not found for testing.")
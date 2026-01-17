import cv2
import moviepy.editor as mp_editor
import numpy as np
import os

def add_text_overlay(frame, text):
    """
    Draws the Viral Headline on a single frame using OpenCV.
    """
    # Make the frame writeable (MoviePy frames are read-only sometimes)
    frame = np.array(frame)
    h, w, _ = frame.shape
    
    # Text Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 4
    color = (255, 255, 0) # Cyan/Yellow (BGR format not RGB for MoviePy? No, MoviePy is RGB)
    # MoviePy uses RGB, so (255, 255, 0) is Yellow.
    
    # Calculate Text Size to Center it
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = 100 # Position near top
    
    # Draw Black Outline (for readability)
    cv2.putText(frame, text, (x, y), font, font_scale, (0,0,0), thickness+6, cv2.LINE_AA)
    # Draw Main Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

def detect_face_center(frame):
    # (Same Face Detection Logic - compacted for brevity)
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            return (x + (w / 2)) / frame.shape[1]
    except: pass
    return None 

def smart_crop_to_vertical(video_path, start_time, end_time, headline, output_path="final_output.mp4"):
    print(f"🎬 Processing Video: {os.path.basename(video_path)}")
    full_clip = None 
    final_clip = None

    try:
        full_clip = mp_editor.VideoFileClip(video_path)
        if end_time > full_clip.duration: end_time = full_clip.duration
        
        # 1. Cut the segment
        clip = full_clip.subclip(start_time, end_time)
        
        # 2. Smart Crop Logic
        frame = clip.get_frame((end_time - start_time) / 2)
        face_x = detect_face_center(frame)
        
        w, h = clip.size
        new_width = int(h * (9/16))
        
        if face_x:
            pixel_x = int(face_x * w)
            x1 = pixel_x - (new_width // 2)
            x2 = pixel_x + (new_width // 2)
            if x1 < 0: x1 = 0; x2 = new_width
            if x2 > w: x2 = w; x1 = w - new_width
        else:
            x1 = (w // 2) - (new_width // 2)
            x2 = (w // 2) + (new_width // 2)

        cropped_clip = clip.crop(x1=x1, y1=0, x2=x2, y2=h).resize(height=1920) 
        
        # 3. APPLY THE TEXT OVERLAY
        # We use .fl_image to apply the OpenCV function to every frame
        print(f"✍️ Burning Headline: '{headline}'")
        final_clip = cropped_clip.fl_image(lambda f: add_text_overlay(f, headline))
        
        final_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", fps=24,
            preset="ultrafast", threads=4, logger=None
        )
        return output_path

    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if full_clip: full_clip.close()
        if final_clip: final_clip.close()
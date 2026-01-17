from google import genai
from google.genai import types
import json
import re

def get_viral_segments(api_key, transcript_text):
    """
    Analyzes transcript to find the TOP 3 viral segments.
    Returns: A list of dictionaries: [{'start': 0, 'end': 60, 'headline': '...', 'reason': '...'}]
    """
    print("🧠 Gemini is analyzing for Top 3 Viral Segments...")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    I have a transcript of a 34-minute video. I need you to identify the TOP 3 most viral, interesting, or high-value segments to turn into Reels/TikToks.
    
    RULES:
    1. Each segment must be between 30 and 60 seconds.
    2. Segments must not overlap.
    3. Write a CLICKBAIT HEADLINE (Max 5 words) for each.
    
    TRANSCRIPT:
    "{transcript_text[:90000]}"  # Limit characters to avoid token limits on very long videos
    
    Output strictly as a JSON LIST of objects:
    [
        {{
            "start_time": 120.5,
            "end_time": 180.0,
            "headline": "THE HIDDEN TRUTH 🤫",
            "reason": "High emotional impact"
        }},
        {{
            "start_time": 500.0,
            "end_time": 560.0,
            "headline": "STOP DOING THIS 🛑",
            "reason": "Controversial advice"
        }}
    ]
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Parse the JSON List
        segments = json.loads(response.text)
        
        # Validate format (sometimes AI returns just one object, wrap it in list if needed)
        if isinstance(segments, dict):
            segments = [segments]
            
        print(f"✨ AI Found {len(segments)} segments.")
        return segments

    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        # Fallback: Return one dummy segment if AI fails
        return [{
            "start_time": 0.0, 
            "end_time": 60.0, 
            "headline": "VIRAL HIGHLIGHT 🔥", 
            "reason": "Error fallback"
        }]
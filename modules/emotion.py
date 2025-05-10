# modules/emotion.py
from deepface import DeepFace
from .utils import capture_frame

def analyze_emotion(camera_url):
    frame = capture_frame(camera_url)
    if frame is None:
        return {"error": "Unable to capture frame for emotion analysis."}

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        return {"emotion": dominant_emotion}
    except Exception as e:
        return {"error": str(e)}

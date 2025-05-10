import cv2
import speech_recognition as sr

# Default fallback URL (can be overwritten by frontend)
camera_url = "http://192.168.0.101:8080/video"

def set_camera_url(url):
    """Set the active camera stream URL (from frontend)."""
    global camera_url
    camera_url = url

def get_camera_url():
    """Return the current active camera stream URL."""
    return camera_url

def capture_frame(url=None):
    """Capture a frame from the camera."""
    if not url:
        url = camera_url
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def listen_command():
    """Listen to user voice input and return the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for command...")
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)
            command = recognizer.recognize_google(audio)
            print(f"Recognized: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError:
            print("Speech recognition service unavailable.")
            return ""

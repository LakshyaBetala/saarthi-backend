from gtts import gTTS
import os
import uuid
from playsound import playsound  # pip install playsound==1.2.2

def speak(text, save_audio=False):
    """Convert text to speech and play it with auto cleanup."""
    audio_path = f"audio_{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)

    if not save_audio:
        try:
            playsound(audio_path)  # ✅ blocks until playback completes
            os.remove(audio_path)
        except Exception as e:
            print(f"Audio error: {e}")
    else:
        return audio_path


def play_audio(audio_path):
    """Play audio based on OS."""
    system = platform.system()
    if system == "Darwin":      # macOS
        os.system(f"afplay {audio_path}")
    elif system == "Windows":
        os.system(f"start {audio_path}")
    else:                       # Linux
        os.system(f"aplay {audio_path}")

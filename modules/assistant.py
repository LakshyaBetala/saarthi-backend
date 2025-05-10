import time
from .audio import speak
from .search import google_search_summary
from .vision import detect_objects_and_direction
from .emotion import analyze_emotion
from .utils import get_camera_url, listen_command
import pywhatkit

def continuous_assistant():
    camera_url = get_camera_url()
    command = listen_command()

    # ✅ Skip empty or failed recognition
    if not command or command.strip() == "":
        return {"message": "No command detected. Waiting..."}

    # ✅ Trigger assistant if name is called
    if any(name in command.lower() for name in ["bharat", "bharath", "barat", "bha"]):
        for name in ["bharat", "bharath", "barat", "bha"]:
            command = command.lower().replace(name, "")
        command = command.strip()

        # 🎯 OBJECT + EMOTION
        if "object" in command or "detect" in command or "ahead" in command:
            result = detect_objects_and_direction(camera_url)

            object_labels = [obj["label"] for obj in result["detected_objects"]]
            if object_labels:
                speak("Objects ahead are " + ", ".join(object_labels))
                time.sleep(4)

            speak(result["direction_suggestion"])
            time.sleep(2)

            emotion_result = analyze_emotion(camera_url)
            if "emotion" in emotion_result:
                emotion = emotion_result["emotion"]
                speak(f"You seem {emotion}")

                if emotion in ["sad", "angry", "fear", "disgust"]:
                    speak("Please take a moment to relax.")
                    speak("Repeating the route: " + result["direction_suggestion"])
            else:
                speak("Could not analyze emotion.")

            return {
                "action": "Object detection + Emotion analysis",
                "objects": object_labels,
                "direction": result["direction_suggestion"],
                "emotion": emotion_result.get("emotion", "unknown")
            }

        # 🔍 SEARCH
        elif "search" in command:
            query = command.replace("search", "").strip()
            speak(f"Searching for {query}")
            return google_search_summary(query)

        # ▶️ PLAY YOUTUBE
        elif "play" in command:
            song = command.replace("play", "").strip()
            speak(f"Playing {song} on YouTube")
            pywhatkit.playonyt(song)
            return {"action": "YouTube", "query": song}

        # 🛑 STOP
        elif "stop" in command or "exit" in command:
            speak("Goodbye!")
            return {"message": "Assistant stopped."}

        # ❓ UNKNOWN
        else:
            speak("Command unclear. Please try again.")
            return {"message": "Unrecognized command."}

    # ❌ Not addressed to assistant
    return {"message": "No assistant trigger word detected."}

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from modules.vision import detect_objects_and_direction
from modules.emotion import analyze_emotion
from modules.audio import speak
from modules.search import google_search_summary
from modules.utils import set_camera_url, get_camera_url
from fastapi.responses import FileResponse

import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https://.*\.vercel\.app|http://localhost:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/set_camera_url")
def set_url(ip: str = Form(...)):
    """Set IP Webcam address (e.g., 192.168.1.5)"""
    url = f"http://{ip}:8080/video"
    set_camera_url(url)
    return {"message": "Camera connected", "stream_url": url}

@app.get("/assistant/listen")
def listen_for_command():
    """Voice-activated assistant interface"""
    from modules.assistant import continuous_assistant
    return continuous_assistant()

@app.get("/detect_objects")
def detect_objects():
    url = get_camera_url()
    return detect_objects_and_direction(url)

@app.get("/analyze_emotion")
def detect_emotion():
    url = get_camera_url()
    return analyze_emotion(url)

@app.post("/search")
def search_query(query: str = Form(...)):
    return google_search_summary(query)


@app.post("/speak")
def speak_text(text: str = Form(...)):
    path = speak(text, save_audio=True)
    return FileResponse(path, media_type="audio/mpeg", filename="output.mp3")


# ✅ Run server using PORT from Render, default to 10000
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

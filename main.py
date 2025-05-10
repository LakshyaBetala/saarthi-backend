# main.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from modules.audio import speak
from modules.search import google_search_summary
from modules.utils import set_camera_url, get_camera_url
from modules.emotion import analyze_emotion
import os, urllib.request
import uvicorn

# ✅ Download YOLOv8 model before import
MODEL_PATH = "models/yolov8n.pt"
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        MODEL_PATH
    )

# ✅ Now safe to import vision
from modules.vision import detect_objects_and_direction

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
    url = f"http://{ip}:8080/video"
    set_camera_url(url)
    return {"message": "Camera connected", "stream_url": url}

@app.get("/assistant/listen")
def listen_for_command():
    from modules.assistant import continuous_assistant
    return continuous_assistant()

@app.get("/detect_objects")
def detect_objects():
    url = get_camera_url()
    return detect_objects_and_direction(url, model_path=MODEL_PATH)

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import uuid
import uvicorn
import imageio.v3 as iio
import cv2

from app.video_parser import VideoParser

PORT = 8080

app = FastAPI()
app.mount("/css", StaticFiles(directory="/src/web_app/css"), name="css")
app.mount("/js", StaticFiles(directory="/src/web_app/js"), name="js")
# app.mount("/mp4", StaticFiles(directory="/src/web_app/mp4"), name="mp4")
# app.mount("/img", StaticFiles(directory="/src/web_app/img"), name="img")

app.video_parser = None
app.frame_idx = 0
app.session_id = ""

@app.on_event("startup")
async def startup_event():
    print("Startup event")
    app.video_parser = VideoParser(device=-1)

@app.get("/", response_class=HTMLResponse)
def get_client_html():
    print("Client web app is launched")
    with open("/src/web_app/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.get("/session", response_class=HTMLResponse)
def session():
    with open("/src/web_app/session.html", "r") as f:
        html_content = f.read()
    return html_content


@app.post("/process_frame")
async def process_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(...),
):
    print("Receiving data from client web app...")
    res = {
        "status": "ok",
        "data": {}
    }

    if session_id != app.session_id:
        print("Session ID not matching")
        res["status"] = "error"
        res["info"] = "Session ID not matching"
        return res

    try:
        frame_bytes = frame.file.read()
        img = iio.imread(frame_bytes, index=None)[...,:3]
    except Exception as e:
        print(f"Failed to parse frames: {e}")
        res["status"] = "error"
        res["info"] = "Failed to parse frames"
        return res

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/src/test.jpg", img)

    app.video_parser.process_next_frame(img, app.frame_idx)
    app.frame_idx += 1

    game = {}
    game["status"] = app.video_parser.status
    res["data"]["game"] = game

    return res

@app.post("/start_session")
async def start_session():
    print("Creating a new game...")

    app.video_parser.reset()

    res = {
        "status": "ok",
        "data": {}
    }

    app.session_id = uuid.uuid4()
    res["data"]["session_id"] = app.session_id

    return res

@app.post("/stop_session")
async def stop_session():
    print("Creating a new game...")

    app.video_parser.reset()

    res = {
        "status": "ok",
        "data": {}
    }

    app.session_id = ""

    return res


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )

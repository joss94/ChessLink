from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import uuid
import uvicorn
import imageio.v3 as iio
import cv2
import numpy as np
import base64
import time

from app.video_parser import VideoParser

from utils.image_utils import make_square_image, crop_board

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
    app.video_parser = VideoParser()

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
    frame: str = Form(...),
):
    start = time.time()
    print("Receiving data from client web app...")
    res = {
        "status": "ok",
        "data": {}
    }

    # if str(session_id) != str(app.session_id):
    #     print(f"Session ID not matching: app: {app.session_id}   web: {session_id}")
    #     res["status"] = "error"
    #     res["info"] = "Session ID not matching"
    #     return res

    try:
        # frame_bytes = frame.file.read()
        # img = iio.imread(frame_bytes, index=None)[...,:3]

        encoded_data = frame.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Failed to parse frames: {e}")
        res["status"] = "error"
        res["info"] = "Failed to parse frames"
        return res

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Frame decoding over... ({time.time() - start})")
    start = time.time()

    app.video_parser.buffer.append(img)
    results = app.video_parser.process_buffer()
    if len(results) < 1:
        print(f"Failed to parse frames, results empty")
        res["status"] = "error"
        res["info"] = "Failed to parse frames"
        return res

    print(f"buffer processed... ({time.time() - start})")
    start = time.time()

    results = results[-1]
    app.frame_idx += 1

    game = {}
    game["status"] = app.video_parser.status

    if len(results["board_poly"]) == 4:
        game["board"] = results["board_poly"].tolist()
        board_mask = cv2.fillPoly(np.zeros((img.shape[0], img.shape[1])), [results["board_poly"]], color=255)
        _, encoded = cv2.imencode('.png', board_mask)
        board_mask_b64 = base64.b64encode(encoded)
        game["board_mask"] = str(board_mask_b64)[2:-1]
    res["data"]["game"] = game


    print(f"Resuls filled... ({time.time() - start})")
    start = time.time()

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
    print("Starting server")
    # while(True):
    #     time.sleep(10)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )

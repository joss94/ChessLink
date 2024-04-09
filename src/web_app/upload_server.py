import json
import uuid
import uvicorn
import cv2
import numpy as np
import base64
import time
import chess
from typing import List, Annotated

from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.video_parser import VideoParser
from utils.image_utils import get_board_image

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


@app.get("/video", response_class=FileResponse)
async def get_video(filename: str):
    return f"/src/web_app/videos/{filename}.mp4"


@app.get("/session", response_class=HTMLResponse)
def session():
    with open("/src/web_app/session.html", "r") as f:
        html_content = f.read()
    return html_content


@app.post("/process_frame")
async def process_frame(
    session_id: Annotated[str, Body()],
    frame: Annotated[list, Body()],
    height: Annotated[int, Body()],
    width: Annotated[int, Body()],
    original_height: Annotated[int, Body()],
    original_width: Annotated[int, Body()],
):
    start = time.time()
    print("Receiving frame from client web app...")
    res = {"status": "ok", "data": {}}

    # if str(session_id) != str(app.session_id):
    #     print(f"Session ID not matching: app: {app.session_id}   web: {session_id}")
    #     res["status"] = "error"
    #     res["info"] = "Session ID not matching"
    #     return res

    try:
        img = np.array(frame, dtype=np.uint8).reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (original_width, original_height))
    except Exception as e:
        print(f"Failed to parse frames: {e}")
        res["status"] = "error"
        res["info"] = "Failed to parse frames"
        return res

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
        board_mask = cv2.fillPoly(
            np.zeros((img.shape[0], img.shape[1])), [results["board_poly"]], color=255
        )
        board_mask = cv2.resize(board_mask, (width, height))
        _, encoded = cv2.imencode(".png", board_mask)
        board_mask_b64 = base64.b64encode(encoded)
        game["board_mask"] = str(board_mask_b64)[2:-1]

        poly = results["board_poly"].astype(np.float64)
        poly[:, 0] *= width / original_width
        poly[:, 1] *= height / original_height
        game["board"] = poly.astype(np.int32).tolist()

        board = chess.Board(results["board"])
        board_png = get_board_image(board, 300)
        board_png = cv2.cvtColor(board_png, cv2.COLOR_BGR2BGRA)
        _, encoded = cv2.imencode(".png", board_png)
        board_png_b64 = base64.b64encode(encoded)
        game["board_png"] = str(board_png_b64)[2:-1]

    res["data"]["game"] = game

    return res


@app.post("/start_session")
async def start_session():
    print("Creating a new game...")

    app.video_parser.reset()

    res = {"status": "ok", "data": {}}

    app.session_id = uuid.uuid4()
    res["data"]["session_id"] = app.session_id

    return res


@app.post("/stop_session")
async def stop_session():
    print("Creating a new game...")

    app.video_parser.reset()

    res = {"status": "ok", "data": {}}

    app.session_id = ""

    return res


if __name__ == "__main__":
    print("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import uvicorn
import imageio.v3 as iio

from app.video_parser import VideoParser

PORT = 8080

app = FastAPI()
app.mount("/css", StaticFiles(directory="/src/web_app/css"), name="css")
app.mount("/js", StaticFiles(directory="/src/web_app/js"), name="js")
# app.mount("/mp4", StaticFiles(directory="/src/web_app/mp4"), name="mp4")
# app.mount("/img", StaticFiles(directory="/src/web_app/img"), name="img")

video_parser = None
frame_idx = 0

@app.on_event("startup")
async def startup_event():
    print("Startup event")
    video_parser = VideoParser(device=-1, yolo_detect=True)

@app.get("/", response_class=HTMLResponse)
def get_client_html():
    print("Client web app is launched")
    with open("/src/web_app/index.html", "r") as f:
        html_content = f.read()
    return html_content


@app.post("/upload")
async def process_video(
    video: UploadFile = File(...),
):
    print("Receiving data from client web app...")

    video_bytes = video.file.read()
    frames = iio.imread(video_bytes, index=None, format_hint=".webm")

    for i in range(frames.shape[0]):
        parser.process_next_frame(frames[i], frame_idx)
        frame_idx += 1
    #     cv2.imwrite(f"/src/{i}.jpg", frames[i])

    return True


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )

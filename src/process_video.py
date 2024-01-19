import os
import cv2

from app.video_parser import VideoParser

DEVICE=1
USE_YOLO=True
# USE_YOLO=False

VIDEO_PATH = "/workspace/ChessLink/data/test_images/carlsen.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/carlsen_2.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/nakamura.mp4"
VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana_1080p.mp4"

if __name__ == "__main__":

    parser = VideoParser(device=DEVICE, yolo_detect=USE_YOLO)

    cap = cv2.VideoCapture(VIDEO_PATH)

    start_idx = 150
    start_idx = 1500
    end_index = start_idx + 20000
    end_index = 1e8

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frame_idx = start_idx
    while(True):
        ret, frame = cap.read()

        if "carlsen" in VIDEO_PATH:
            h, w, _ = frame.shape
            frame[:int(h/2),:,:] = 0

        parser.process_next_frame(frame, frame_idx)

        frame_idx += 1
        if frame_idx > end_index:
            break

    # Process remaining buffer
    self.process_buffer()

    # Release OpenCV objects
    cap.release()
    if self.writer:
        self.writer.release()

    # parser.process_single_image("/workspace/ChessLink/data/gamestate_test/0175.png")

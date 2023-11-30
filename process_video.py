import os
import cv2
import numpy as np

from networks.detection.train import DetectNet
from networks.segmentation.train_segmentation import SegmentNet
import time
import json
import chess
from shapely.geometry import Polygon, Point

from ultralytics import YOLO
import mediapipe as mp
from PIL import Image
import scipy

import math

from chessboard_parser import ChessboardParser
from utils import draw_results_on_image

DEVICE=1
USE_YOLO=True
# USE_YOLO=False

VIDEO_PATH = "/workspace/ChessLink/data/test_images/carlsen.mp4"
VIDEO_PATH = "/workspace/ChessLink/data/test_images/carlsen_2.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/nakamura.mp4"
VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana_1080p.mp4"

class VideoParser():

    def __init__(self, device=3, yolo_detect=False, verbose=True):
        self.chessboard_parser = ChessboardParser(device, yolo_detect)

        self.verbose = verbose

        self.buffer = []
        self.buff_size = 1
        self.writer = None
        self.process_freq = 5
        self.save_individuals = True

        self.hands_net = mp.solutions.hands.Hands(max_num_hands=6, min_detection_confidence=0.1, static_image_mode=True)
        self.mask = None
        self.time_since_refresh=1000

        self.board = chess.Board()
        self.safe_board = chess.Board()
        self.fen = ""
        self.unchanged_pos = 0
        self.board_img = None

        self.board_stability=3


    def compute_accuracy(self, gt_pos):
        if not gt_pos:
            return [], 0.0
        pos = self.board.fen()
        different_squares = [(7 - int(i/8))*8 + i%8 for i in range(64) if pos[i] != gt_pos[i]]
        acc = (64 - len(different_squares)) / 64
        return different_squares, acc


    def process_buffer(self):

        if len(self.buffer) == 0:
            return

        results = self.chessboard_parser.process_images([f for _, f in self.buffer])

        for (r, (frame_idx, image)) in zip(results, self.buffer):

            if r["info"] == "ok":

                self.board.set_board_fen(r["board"])
                if self.board.fen() == self.fen:
                    self.unchanged_pos += 1
                else:
                    self.unchanged_pos = 0

                if self.unchanged_pos > self.board_stability:
                    if self.board.fen() != self.safe_board.fen():
                        self.safe_board = self.board.copy()
                        if self.verbose:
                            print("---------------")
                            print(self.safe_board)
                            print("---------------")

                self.fen = self.board.fen()

            draw_results_on_image(image, r)

            if self.save_individuals:
                if not os.path.exists("output"):
                    os.makedirs("output", exist_ok=True)
                cv2.imwrite(f"output/{frame_idx}.jpg", image)
            else:
                if not writer:
                    vid_size = (1080,720)
                    fps = int(25/self.process_freq)
                    writer = cv2.VideoWriter("output/video.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, vid_size)
                writer.write(cv2.resize(image, vid_size))

        self.buffer.clear()

    def process_video(self, video_path):

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        start_idx = 2500
        end_index = start_idx + 20000

        frame_idx = start_idx
        while(ret):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if "carlsen" in video_path:
                h, w, _ = frame.shape
                frame[:int(h/2),:,:] = 0

            self.time_since_refresh += 1

            if ret:

                frame_idx += self.process_freq
                # frame = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))
                self.buffer.append([frame_idx, frame])

                # Process only one of every 3 frame, and delay for 15 frames after successful parsing
                if len(self.buffer) == self.buff_size:
                    self.process_buffer()

            else:
                print("COULD NOT READ FRAME, END OF VIDEO FILE?")
                break

            if frame_idx > end_index:
                break

        # Process remaining buffer
        self.process_buffer()

        cap.release()
        if self.writer:
            self.writer.release()



if __name__ == "__main__":
    with(open("/workspace/CL/data/gt.json"))as f:
        gt = json.loads(f.read())["positions"]

    parser = VideoParser(device=DEVICE, yolo_detect=USE_YOLO)
    parser.process_video(VIDEO_PATH)
    # parser.process_single_image("/workspace/ChessLink/data/gamestate_test/0175.png")

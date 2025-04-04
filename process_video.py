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
from chess_utils.chess_utils import *

DEVICE=1
USE_YOLO=True
# USE_YOLO=False
STILL_TIME = 10

VIDEO_PATH = "/workspace/ChessLink/data/test_images/carlsen.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/carlsen_2.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/nakamura.mp4"
VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana.mp4"
# VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana_1080p.mp4"

class VideoParser():

    """
    Constructor
    """
    def __init__(self, device=3, yolo_detect=False, verbose=True):
        self.chessboard_parser = ChessboardParser(device, yolo_detect)

        self.verbose = verbose

        self.buffer = []
        self.buff_size = 1
        self.writer = None
        self.process_freq = 1
        self.save_individuals = False
        self.save_video = False

        self.safe_board = None
        self.safe_board = chess.Board(chess.STARTING_FEN)
        self.board_img = None
        self.pieces = []
        self.board_mask = None
        self.last_frame = None
        self.still = 0


    """
    Computes a motion mask that extracts areas of the scene that have changed between
    2 frames. It is possible to provide a mask to run the motion detection only on a
    specific subregion of the scene
    """
    def compute_motion_mask(self, frame1, frame2, mask = None):
        img1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # compute grayscale blurred image difference
        frame_diff = cv2.absdiff(img2, img1)
        frame_diff = cv2.medianBlur(frame_diff, 3)
        if mask is not None:
            frame_diff = frame_diff * mask

        motion_mask = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)[1]
        motion_mask = cv2.medianBlur(motion_mask, 3)

        # morphological operations
        kernel=np.array((9,9))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        move_count = np.sum(motion_mask) / 255

        return motion_mask, move_count


    """
    Computes the intersection over union metric for 2 bounding boxes.
    This is useful when trying to determine if a detected piece has moved or not
    """
    def compute_iou(self, boxA, boxB):
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if boxAArea <= 0 and boxBArea <= 0:
            return 0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        return interArea / float(boxAArea + boxBArea - interArea)


    """
    Run analysis on the buffer.
    There are 5 steps:
        1. Predict "raw" chess position for each image of the buffer
        2. Use previous board state to identify moves leading to the new positions, while possibly
           correcting the "raw" chess predictions to stay within plausible game moves
        3. Update current state (stable board position, board mask, bounding boxes, ...)
        4. Draw results on images and save them if necessary
        5. Clear the buffer
    """
    def process_buffer(self):

        if len(self.buffer) == 0:
            return None

        # Parse positions frame by frame
        results = self.chessboard_parser.process_images([f for _, f in self.buffer])

        # Handle results to update board with "temporal consistency"
        for (r, (frame_idx, image)) in zip(results, self.buffer):

            # Ignore frames where board was not detected, or hand was in front
            if r["info"] != "ok":
                continue

            board = chess.Board(r["board"])

            # If stable board has not been initialized, then start it with the first position
            # that was parsed
            if self.safe_board is None:
                self.safe_board = board.copy()
                self.safe_board.set_castling_fen("KQkq")
                self.pieces = r["pieces"]
                continue

            # find pieces that did not move, to speed up search of possible moves:
            still_squares = []
            for i, piece in enumerate(r["pieces"]):
                for prev_piece in self.pieces:
                    # Empty squares means a move could have happened there (missed detection)
                    if not piece["piece"] or not prev_piece["piece"]:
                        continue

                    # Different piece label means a move could have happened there (take)
                    if piece["piece"] != prev_piece["piece"]:
                        continue

                    # Different IOU means a move coudl have happened there (back and forth)
                    if abs(self.compute_iou(piece["box"], prev_piece["box"]) - 1.0) > 0.1:
                        continue

                    # Otherwise, this piece stood still, no moved happened here
                    still_squares.append(i)

            # Try to find moves between positions only if some still squares have been found,
            # otherwise it is too heavy computationally
            moves_list = []
            if len(still_squares) > 0:
                moves_list = moves_between_positions(self.safe_board, board, 5, False, still_squares)

            # Update stable board with identified moves
            if len(moves_list):
                for m in moves_list[0][0]:
                    self.safe_board.push(m)

                print(board_pgn_string(self.safe_board))
                print(boards_to_string([self.safe_board, board]))

            elif self.safe_board.fen().split(" ")[0] != board.fen().split(" ")[0]:
                print("COULD NOT FIND MOVES BETWEEN THOSE BOARDS: ")
                print(boards_to_string([self.safe_board, board]))

            # Update pieces detections
            self.pieces = r["pieces"]

            # Update board detection
            board_poly = r["board_poly"]
            self.board_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(self.board_mask, [board_poly], color = 1)
            self.board_mask = cv2.dilate(self.board_mask, None, iterations=50)


        # Draw results on images
        if self.save_individuals or self.save_video:
            for (r, (frame_idx, image)) in zip(results, self.buffer):
                image_cpy = image.copy()
                draw_results_on_image(image_cpy, r)

                if self.save_individuals:
                    if not os.path.exists("output"):
                        os.makedirs("output", exist_ok=True)
                    cv2.imwrite(f"output/{frame_idx}.jpg", image_cpy)

                if self.save_video:
                    if not writer:
                        vid_size = (1080,720)
                        fps = int(25/self.process_freq)
                        writer = cv2.VideoWriter("output/video.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, vid_size)
                    writer.write(cv2.resize(image_cpy, vid_size))

        # Clear buffer
        self.buffer.clear()

        return results


    def process_next_frame(self, frame, frame_idx):

        # First thing we need is to detect a board in the image
        if self.board_mask is None:
            self.buffer.append([frame_idx, frame])

        # If we have more than one frame, try to detect motion and run analysis to one
        # "still" image after each motion
        elif self.last_frame is not None:

            _, move_count = self.compute_motion_mask(frame, self.last_frame, self.board_mask)

            self.still = self.still + 1 if move_count == 0 else 0
            if self.still == STILL_TIME:
                self.buffer.append([frame_idx, frame])

        # Update last frame reference
        self.last_frame = frame

        # Run analysis on buffer when it is full
        if len(self.buffer) >= self.buff_size:
            results = self.process_buffer()




if __name__ == "__main__":
    with(open("/workspace/CL/data/gt.json"))as f:
        gt = json.loads(f.read())["positions"]

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

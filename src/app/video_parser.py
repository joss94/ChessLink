import cv2
import numpy as np
import time

import chess

from app.chessboard_parser import ChessboardParser
from utils.image_utils import draw_results_on_image
from utils.chess_utils import *

class VideoParser():

    """
    Constructor
    """
    def __init__(self, device=-1, verbose=True):
        self.chessboard_parser = ChessboardParser(device)

        self.verbose = verbose
        self.buff_size = 1
        self.process_freq = 1

        self.save_individuals = False
        self.save_video = False
        self.still_time = 10

        self.reset()

    def reset(self):
        self.buffer = []
        self.writer = None
        self.safe_board = None
        self.safe_board = chess.Board(chess.STARTING_FEN)
        self.board_img = None
        self.pieces = []
        self.board_mask = None
        self.last_frame = None
        self.still = 0
        self.status = ""
        self.last_board_det_time = -1


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
        for (r, (frame_idx, frame)) in zip(results, self.buffer):

            self.status = r["status"]

            # Ignore frames where board was not detected, or hand was in front
            if r["status"] != "OK":
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
                moves_list = moves_between_positions(self.safe_board, board, 2, False, still_squares)

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


        # # Draw results on images
        # if self.save_individuals or self.save_video:
        #     for (r, (frame_idx, image)) in zip(results, self.buffer):
        #         image_cpy = image.copy()
        #         draw_results_on_image(image_cpy, r)

        #         if self.save_individuals:
        #             if not os.path.exists("output"):
        #                 os.makedirs("output", exist_ok=True)
        #             cv2.imwrite(f"output/{frame_idx}.jpg", image_cpy)

        #         if self.save_video:
        #             if not writer:
        #                 vid_size = (1080,720)
        #                 fps = int(25/self.process_freq)
        #                 writer = cv2.VideoWriter("output/video.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, vid_size)
        #             writer.write(cv2.resize(image_cpy, vid_size))

        # Clear buffer
        self.buffer.clear()

        return results


    def process_next_frame(self, frame, frame_idx):

        # First thing we need is to detect a board in the image
        t = time.time()
        if t - self.last_board_det_time > 5:
            self.last_board_det_time = t
            self.buffer.append([frame_idx, frame])
            self.process_buffer()

        # If we have more than one frame, try to detect motion and run analysis to one
        # "still" image after each motion
        if self.board_mask is not None and self.last_frame is not None:

            _, move_count = self.compute_motion_mask(frame, self.last_frame, self.board_mask)

            self.still = self.still + 1 if move_count == 0 else 0
            if self.still == self.still_time:
                self.buffer.append([frame_idx, frame])

        # Update last frame reference
        self.last_frame = frame

        # Run analysis on buffer when it is full
        if len(self.buffer) >= self.buff_size:
            results = self.process_buffer()



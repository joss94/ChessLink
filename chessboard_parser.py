import cv2
import numpy as np

from networks.detection.train import DetectNet
from networks.segmentation.train_segmentation import SegmentNet
import time
import json
import chess
import chess.svg
from cairosvg import svg2png
from shapely.geometry import Polygon, Point

from ultralytics import YOLO
import mediapipe as mp
from io import BytesIO
from PIL import Image
import scipy

import math
from utils import make_square_image, crop_board

class ChessboardParser():

    def __init__(self, device=3, yolo_detect=False):
        self.class_names = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

        self.device_number = device
        if device < 0:
            self.device="cpu"
        else:
            self.device=f"cuda:{device}"

        self.yolo_detect=yolo_detect

        self.segment_net = YOLO('/workspace/ChessLink/runs/segment/train7/weights/best.pt')
        # self.segment_net = SegmentNet(pretrained_path="/workspace/CL/model/segment/best_ref.torch")

        if self.yolo_detect:
            self.detect_net = YOLO('/workspace/ChessLink/runs/detect/train45/weights/last.pt')
        else:
            self.detect_net = DetectNet(pretrained_path="/workspace/ChessLink/model/detection/latest.torch", device=self.device)

        self.hands_net = mp.solutions.hands.Hands(max_num_hands=6, min_detection_confidence=0.1, static_image_mode=True)
        self.mask = None

        self.hands_on_board = 0
        self.fen = ""

        # Match pieces to squares
        self.pieces_labels = [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ]

    def extract_squares_coords(self, image, mask):

        h = image.shape[0]
        w = image.shape[1]

        self.mask = mask

        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)

        if len(c) < 4:
            print("FOUND CONTOUR WITH LENGTH < 4")
            return None

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)

        # Simplify the contour to a 4-sides polygon.
        eps_factor = 0.002
        epsilon = eps_factor*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)

        while len(approx) > 4:
            eps_factor = eps_factor + 0.002
            epsilon = eps_factor*cv2.arcLength(approx,True)
            approx = cv2.approxPolyDP(approx,epsilon,True)
            if eps_factor >= 0.05:
                return None

        if len(approx) != 4:
            print("FAILED TO APPROXIMATE CONTOUR WITH 4 POINTS")
            return None

        approx = np.reshape(approx, (4,2))

        # ------------------------------------------------------------------------------------------------
        # This polygon is not very precise because
        # the simplification does not minimize the error. We can find better lines by using this
        # polygon approximation to group the contour points into 4 sides, and then interpolate
        # the 2D data points of each side to an optimized line equation
        #------------------------------------------------------------------------------------------------

        # Make sure the polygon coords are clockwise
        center = np.mean(approx, axis=0)
        approx = sorted(approx, key=lambda a: -1.0 * (np.arctan2(a[0] - center[0], a[1] - center[1]) + 2*3.1415))
        approx = np.roll(approx, 1, axis=0)

        # Cluster the initial contour point to 4 lines
        lines = np.array([
            [approx[0], approx[1]],
            [approx[1], approx[2]],
            [approx[2], approx[3]],
            [approx[3], approx[0]],
        ])

        pts_by_line = [[], [], [], []]
        for pt in c:
            pt = np.array(pt[0])
            min_index = -1
            min_dist = -1
            for i, line in enumerate(lines):
                d = np.linalg.norm(np.cross(line[1]-line[0], line[0]-pt))/np.linalg.norm(line[1]-line[0])
                if d < min_dist or min_dist < 0:
                    min_index = i
                    min_dist = d
            pts_by_line[min_index].append(pt)

        # Extract lines coefficients from clustered points
        lines_coeffs = []
        for cluster in pts_by_line:
            cluster = np.array(cluster)
            coeffs = np.polyfit(cluster[:,0],cluster[:,1],1)
            lines_coeffs.append(coeffs)

        # Compute lines intersections to get the corners
        def seg_intersect(coeffs1, coeffs2):
            x = (coeffs2[1] - coeffs1[1]) / (coeffs1[0] - coeffs2[0])
            y = -(coeffs1[1] * coeffs2[0] - coeffs2[1] * coeffs1[0]) / (coeffs1[0] - coeffs2[0])
            return [x, y]

        real_corners = [seg_intersect(lines_coeffs[i], lines_coeffs[(i+1)%4]) for i in range(4)]
        real_corners = np.array(real_corners)

        square_size = min(w, h)
        chessboard_squares = []
        for r in range(9):
            for c in range(9):
                pt = np.array([w/2+r-5, h/2+c-5]).astype(np.float64) * square_size/8
                chessboard_squares.append(pt)
        chessboard_squares = np.array(chessboard_squares, np.int32)

        chessboard_corners = np.array([
            chessboard_squares[72],
            chessboard_squares[80],
            chessboard_squares[8],
            chessboard_squares[0],
        ], np.int32)

        M, _ = cv2.findHomography(chessboard_corners.reshape(-1,1,2), real_corners.reshape(-1,1,2))

        real_squares = cv2.perspectiveTransform(np.float32(chessboard_squares).reshape(-1,1,2), M).reshape(-1,2)

        real_squares_corners = []
        for r in range(8):
            for c in range(8):
                c1 = r * 9 + c
                c2 = c1 + 1
                c3 = c1 + 9
                c4 = c3 + 1

                real_squares_corners.append(np.array([real_squares[c] for c in [c1, c2, c4, c3]], np.int32))
        real_squares_corners = np.int32(real_squares_corners)


        board_poly = np.array([
            real_squares_corners[7][1],
            real_squares_corners[63][2],
            real_squares_corners[56][3],
            real_squares_corners[0][0]]
            )


        board_mask = cv2.fillPoly(np.zeros(self.mask.shape, np.uint8), [board_poly], color=255)
        board_mask = cv2.bitwise_xor(self.mask, board_mask)

        mask_overlay = np.sum(board_mask.reshape(-1)) / np.sum(self.mask.reshape(-1))

        real_squares_corners = np.float32(real_squares_corners)
        real_squares_corners[:,:,0] *= w / self.mask.shape[1]
        real_squares_corners[:,:,1] *= h / self.mask.shape[0]
        real_squares_corners = np.int32(real_squares_corners)

        self.mask = cv2.resize(self.mask, (w,h))

        if mask_overlay > 0.1:
            return None

        return real_squares_corners

    def extract_position(self, image, detections, squares_corners):

        boxes, scores, labels = detections

        h = image.shape[0]
        w = image.shape[1]

        real_square_centers = np.mean(squares_corners, axis=1, dtype=np.int32)
        pieces = [
            {
                "piece": None,
                "score": -1.0,
                "box": [],
                "center": real_square_centers[i],
            } for i in range(64)
        ]

        squares_polygons = [Polygon(squares_corners[sq]) for sq in range(64)]

        board = chess.Board.empty()

        highest_black_piece_square=None
        highest_black_piece_height=0
        highest_white_piece_square=None
        highest_white_piece_height=0

        for box, score, label in zip(boxes, scores, labels):

            label -= 1

            # Build a square polygon with same width as box, bottom-aligned with the box
            # This polygon represents the "foot" of the piece, discarding the top part which
            # can be superimposed to other squares
            box_base = np.int32([
                [box[0]*w, (box[3] - 0.5 * (box[2] - box[0]))*h],
                [box[2]*w, (box[3] - 0.5 * (box[2] - box[0]))*h],
                [box[2]*w, box[3]*h],
                [box[0]*w, box[3]*h]
            ])

            box_poly = Polygon(box_base)

            # Find the square that maximizes the intersection with the piece "foot"
            intersections = [(sq, squares_polygons[sq].intersection(box_poly).area) for sq in range(64)]
            best_square, area = max(intersections, key=lambda x: x[1])

            piece = chess.Piece(piece_type = self.pieces_labels[(label)%6], color = label>=6)

            if area > 0 and score > pieces[best_square]["score"]:
                pieces[best_square]["score"] = float(score.cpu())
                pieces[best_square]["box"] = box
                pieces[best_square]["piece"] = piece.symbol()
                board.set_piece_at(best_square, piece)


        return board, pieces

    def process_images(self, images):

        # Adjust aspect ratio to fit train set
        squared_images = [make_square_image(img) for img in images]

        # Run segmentation and detection
        masks_batch = self.segment_net(squared_images, device=self.device, verbose=False)

        results = []

        for image_idx, image in enumerate(squared_images):

            result = {
                "pieces": [],
                "board": "",
                "info": "ok"
                }

            h, w, _ = image.shape

            if not masks_batch[image_idx].masks:
                continue

            mask = np.uint8(masks_batch[image_idx].masks.data.cpu().numpy() * 255)
            mask = np.swapaxes(mask, 0, 1)
            mask = np.swapaxes(mask, 1, 2)
            if mask.shape[2] > 1:
                mask = np.expand_dims(mask[:,:,0],2)

            squares_corners = self.extract_squares_coords(image, mask)

            result["squares"] = squares_corners

            if squares_corners is None:
                result["info"] = "COULD NOT FIND BOARD"

            else:

                mask = cv2.resize(mask, (w,h))

                board_poly = np.array([
                    squares_corners[7][1],
                    squares_corners[63][2],
                    squares_corners[56][3],
                    squares_corners[0][0]]
                 )

                board_mask = cv2.fillPoly(np.zeros(mask.shape, np.uint8), [board_poly], color=255)

                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                imageRGB.flags.writeable=False
                hands_dets = self.hands_net.process(imageRGB)

                if hands_dets.multi_hand_landmarks:
                    hands_mask = np.zeros((h,w))
                    for handLms in hands_dets.multi_hand_landmarks: # working with each hand
                        for _, lm in enumerate(handLms.landmark):
                            h, w, _ = image.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(hands_mask, (cx, cy), int(0.01 * w), 255, cv2.FILLED)
                            cv2.circle(image, (cx, cy), int(0.01 * w), (0,0,255), cv2.FILLED)

                    board_mask = board_mask/2 + hands_mask/2
                    board_mask[board_mask<200] = 0

                    if np.max(board_mask.reshape(-1)) > 0:
                        self.hands_on_board = min(self.hands_on_board+1, 3)
                    else:
                        self.hands_on_board = max(self.hands_on_board-1, 0)
                else:
                    self.hands_on_board = max(self.hands_on_board-1, 0)

                if self.hands_on_board > 0:
                    result["info"] = "HAND ON BOARD"

                else:
                    img_cropped, [X, Y, W, H] = crop_board(image, board_poly)
                    img_cropped_flipped = cv2.flip(img_cropped, 1)

                    if self.yolo_detect:
                        output = self.detect_net([img_cropped, img_cropped_flipped], device = [self.device_number], verbose=False, conf=.2)

                        detections = output[0]
                        detections_flipped = output[1]

                        boxes = np.array([b.cpu().numpy() for b in detections.boxes.xyxyn]).reshape((-1,4))
                        labels = [int(c.cpu()) + 1 for c in detections.boxes.cls]
                        scores = [s.cpu() for s in detections.boxes.conf]

                        boxes_flipped = np.array([b.cpu().numpy() for b in detections_flipped.boxes.xyxyn]).reshape((-1,4))
                        labels_flipped = [int(c.cpu()) + 1 for c in detections_flipped.boxes.cls]
                        scores_flipped = [s.cpu() for s in detections_flipped.boxes.conf]
                        boxes_flipped[:, [0, 2]]*=-1
                        boxes_flipped[:, [0, 2]]+=1
                        boxes_flipped[:, [2, 0]] = boxes_flipped[:, [0, 2]]

                        boxes = np.concatenate([boxes, boxes_flipped])
                        scores.extend(scores_flipped)
                        labels.extend(labels_flipped)
                    else:
                        boxes, scores, labels = self.detect_net.infer([img_cropped], 0.5)[0]

                    for box in boxes:
                        box[0] = (box[0] * W + X) / w
                        box[2] = (box[2] * W + X) / w
                        box[1] = (box[1] * H + Y) / h
                        box[3] = (box[3] * H + Y) / h


                    detections = (boxes, scores, labels)
                    board, pieces = self.extract_position(image, detections, squares_corners)

                    original_h, original_w, _ = images[image_idx].shape
                    dw = w - original_w
                    dh = h - original_h

                    for p in pieces:
                        if not p["piece"]:
                            continue
                        p["box"][0] = (p["box"][0] * w - 0.5 * dw) / original_w
                        p["box"][2] = (p["box"][2] * w - 0.5 * dw) / original_w
                        p["box"][1] = (p["box"][1] * h - 0.5 * dh) / original_h
                        p["box"][3] = (p["box"][3] * h - 0.5 * dh) / original_h

                        p["center"][0] -= int(0.5 * dw)
                        p["center"][1] -= int(0.5 * dh)


                    for sq in result["squares"]:
                        for pt in sq:
                            pt[0] -= int(0.5 * dw)
                            pt[1] -= int(0.5 * dh)

                    result["pieces"] = pieces
                    result["board"] = board.fen().split(" ")[0]

            results.append(result)

        return results
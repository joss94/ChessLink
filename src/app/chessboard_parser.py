import cv2
import numpy as np
from pathlib import Path

import time
import chess
import chess.svg
from shapely.geometry import Polygon

from networks.detection.YOLO_det_onnx import YOLOv8ONNX, YOLOv8SegONNX
from ultralytics import YOLO
from utils.image_utils import make_square_image, crop_board

DETECT_WEIGHTS = str(Path("../src/model/detection/weights.onnx"))
SEGMENT_WEIGHTS = str(Path("./model/segment/weights.onnx"))
HANDS_WEIGHTS = str(Path("./model/yolov8n-seg.pt"))

CONF = 0.5

class ChessboardParser:
    def __init__(self, detect_weights=DETECT_WEIGHTS, segment_weights=SEGMENT_WEIGHTS, device="cpu"):
        self.class_names = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]

        self.device = device
        self.detect_weights = detect_weights
        self.segment_weights = segment_weights
        self.hands_weights = HANDS_WEIGHTS

        if HANDS_WEIGHTS.endswith("onnx"):
            self.hand_net = YOLOv8SegONNX(HANDS_WEIGHTS)
        else:
            self.hand_net = YOLO(HANDS_WEIGHTS)

        if segment_weights.endswith("onnx"):
            self.segment_net = YOLOv8SegONNX(segment_weights)
        else:
            self.segment_net = YOLO(segment_weights)

        if detect_weights.endswith("onnx"):
            self.detect_net = YOLOv8ONNX(detect_weights)
        else:
            self.detect_net = YOLO(detect_weights)

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

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            print("DID NOT FIND ANY CONTOUR")
            return None

        c = max(contours, key=cv2.contourArea)

        if len(c) < 4:
            print("FOUND CONTOUR WITH LENGTH < 4")
            return None

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)

        # Simplify the contour to a 4-sides polygon.
        eps_factor = 0.002
        epsilon = eps_factor * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        while len(approx) > 4:
            eps_factor = eps_factor + 0.002
            epsilon = eps_factor * cv2.arcLength(approx, True)
            approx = cv2.approxPolyDP(approx, epsilon, True)
            if eps_factor >= 0.05:
                print(f"eps factor too high")
                return None

        if len(approx) != 4:
            print("FAILED TO APPROXIMATE CONTOUR WITH 4 POINTS")
            return None

        approx = np.reshape(approx, (4, 2))

        # ------------------------------------------------------------------------------------------------
        # This polygon is not very precise because
        # the simplification does not minimize the error. We can find better lines by using this
        # polygon approximation to group the contour points into 4 sides, and then interpolate
        # the 2D data points of each side to an optimized line equation
        # ------------------------------------------------------------------------------------------------

        # Make sure the polygon coords are clockwise
        center = np.mean(approx, axis=0)
        approx = sorted(
            approx,
            key=lambda a: -1.0
            * (np.arctan2(a[0] - center[0], a[1] - center[1]) + 2 * 3.1415),
        )
        approx = np.roll(approx, 1, axis=0)

        # Cluster the initial contour point to 4 lines
        lines = np.array(
            [
                [approx[0], approx[1]],
                [approx[1], approx[2]],
                [approx[2], approx[3]],
                [approx[3], approx[0]],
            ]
        )

        pts_by_line = [[], [], [], []]
        for pt in c:
            pt = np.array(pt[0])
            min_index = -1
            min_dist = -1
            for i, line in enumerate(lines):
                d = np.linalg.norm(
                    np.cross(line[1] - line[0], line[0] - pt)
                ) / np.linalg.norm(line[1] - line[0])
                if d < min_dist or min_dist < 0:
                    min_index = i
                    min_dist = d
            pts_by_line[min_index].append(pt)

        # Extract lines coefficients from clustered points
        lines_coeffs = []
        for cluster in pts_by_line:
            cluster = np.array(cluster)

            # If one of the lines has less than 2 points, extraction will fail
            if cluster.shape[0] < 2:
                return None
            coeffs = np.polyfit(cluster[:, 0], cluster[:, 1], 1)
            lines_coeffs.append(coeffs)

        # Compute lines intersections to get the corners
        def seg_intersect(coeffs1, coeffs2):
            x = (coeffs2[1] - coeffs1[1]) / (coeffs1[0] - coeffs2[0])
            y = -(coeffs1[1] * coeffs2[0] - coeffs2[1] * coeffs1[0]) / (
                coeffs1[0] - coeffs2[0]
            )
            return [x, y]

        real_corners = [
            seg_intersect(lines_coeffs[i], lines_coeffs[(i + 1) % 4]) for i in range(4)
        ]
        real_corners = np.array(real_corners)

        square_size = min(w, h)
        chessboard_squares = []
        for r in range(9):
            for c in range(9):
                pt = (
                    np.array([w / 2 + r - 5, h / 2 + c - 5]).astype(np.float64)
                    * square_size
                    / 8
                )
                chessboard_squares.append(pt)
        chessboard_squares = np.array(chessboard_squares, np.int32)

        chessboard_corners = np.array(
            [
                chessboard_squares[72],
                chessboard_squares[80],
                chessboard_squares[8],
                chessboard_squares[0],
            ],
            np.int32,
        )

        M, _ = cv2.findHomography(
            chessboard_corners.reshape(-1, 1, 2), real_corners.reshape(-1, 1, 2)
        )

        real_squares = cv2.perspectiveTransform(
            np.float32(chessboard_squares).reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        real_squares_corners = []
        for r in range(8):
            for c in range(8):
                c1 = r * 9 + c
                c2 = c1 + 1
                c3 = c1 + 9
                c4 = c3 + 1

                real_squares_corners.append(
                    np.array([real_squares[c] for c in [c1, c2, c4, c3]], np.int32)
                )
        real_squares_corners = np.int32(real_squares_corners)

        board_poly = np.array(
            [
                real_squares_corners[7][1],
                real_squares_corners[63][2],
                real_squares_corners[56][3],
                real_squares_corners[0][0],
            ]
        )

        board_mask = cv2.fillPoly(
            np.zeros(mask.shape, np.uint8), [board_poly], color=255
        )
        board_mask = cv2.bitwise_xor(mask, board_mask)

        mask_overlay = np.sum(board_mask.reshape(-1)) / np.sum(mask.reshape(-1))

        real_squares_corners = np.float32(real_squares_corners)
        real_squares_corners[:, :, 0] *= w / mask.shape[1]
        real_squares_corners[:, :, 1] *= h / mask.shape[0]
        real_squares_corners = np.int32(real_squares_corners)

        # if mask_overlay > 0.1:
        #     print("TOO BIG MASK OVERLAY")
        #     return None

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
            }
            for i in range(64)
        ]

        squares_polygons = [Polygon(squares_corners[sq]) for sq in range(64)]

        board = chess.Board.empty()

        for box, score, label in zip(boxes, scores, labels):

            # Build a square polygon with same width as box, bottom-aligned with the box
            # This polygon represents the "foot" of the piece, discarding the top part which
            # can be superimposed to other squares
            box_base = np.int32(
                [
                    [box[0] * w, (box[3] - 0.5 * (box[2] - box[0])) * h],
                    [box[2] * w, (box[3] - 0.5 * (box[2] - box[0])) * h],
                    [box[2] * w, box[3] * h],
                    [box[0] * w, box[3] * h],
                ]
            )

            box_poly = Polygon(box_base)

            # Find the square that maximizes the intersection with the piece "foot"
            intersections = [
                (sq, squares_polygons[sq].intersection(box_poly).area)
                for sq in range(64)
            ]
            best_square, area = max(intersections, key=lambda x: x[1])

            piece = chess.Piece(
                piece_type=self.pieces_labels[(label) % 6], color=label >= 6
            )

            if area > 0 and score > pieces[best_square]["score"]:
                pieces[best_square]["score"] = score
                pieces[best_square]["box"] = box
                pieces[best_square]["piece"] = piece.symbol()
                board.set_piece_at(best_square, piece)

        return board, pieces

    def process_images(self, images):

        squared_images = [make_square_image(img)[0] for img in images]

        # Run segmentation and detection
        t = time.time()
        board_masks = []
        for image in squared_images:
            if(self.segment_weights.endswith("onnx")): # ONNX
                boxes, segments, masks = self.segment_net(image)
                if len(masks) == 0:
                    masks = np.zeros((640, 640, 1))
            else: # Pytorch
                masks = self.segment_net(image, verbose=False)
                if masks[0].masks is not None:
                    masks = masks[0].masks.data.cpu().numpy()
                    masks = np.swapaxes(np.swapaxes(masks, 0, 1), 1, 2)
                else:
                    masks = np.zeros((640, 640, 1))

            board_mask = np.uint8(masks * 255)[..., 0]
            board_masks.append(board_mask)

        print(f"{time.time() - t} - Running segmentation network")
        t = time.time()

        results = []

        for original, image, mask in zip(images, squared_images, board_masks):

            result = {"pieces": [], "board": "", "status": "OK", "board_poly": []}

            mask = np.expand_dims(mask, 2)

            squares_corners = self.extract_squares_coords(image, mask)
            if squares_corners is None:
                result["status"] = "BOARD_NOT_FOUND"
                results.append(result)
                continue

            result["squares"] = squares_corners

            h, w, _ = image.shape
            dw = w - original.shape[1]
            dh = h - original.shape[0]

            mask = cv2.resize(mask, (w, h))

            board_poly = np.array(
                [
                    squares_corners[7][1],
                    squares_corners[63][2],
                    squares_corners[56][3],
                    squares_corners[0][0],
                ]
            )
            result["board_poly"] = board_poly

            board_mask = cv2.fillPoly(
                np.zeros(mask.shape, np.uint8), [board_poly], color=255
            )

            if(self.hands_weights.endswith("onnx")): # ONNX
                boxes, segments, masks = self.hand_net(image)
                if len(masks) == 0:
                    masks = np.zeros((640, 640, 1))
            else: #Pytorch
                masks = self.hand_net(image, verbose = False)
                if masks[0].masks is not None:
                    masks = masks[0].masks.data.cpu().numpy()
                    masks = np.swapaxes(np.swapaxes(masks, 0, 1), 1, 2)
                else:
                    masks = np.zeros((640, 640, 1))

            hands_mask = np.uint8(masks * 255)[..., 0]
            hands_mask = cv2.resize(hands_mask, (w, h))

            mixed_mask = board_mask / 2 + hands_mask / 2
            mixed_mask[mixed_mask < 200] = 0

            if np.max(mixed_mask.reshape(-1)) > 0:
                self.hands_on_board = 1  # min(self.hands_on_board+1, 3)
            else:
                self.hands_on_board = 0  # max(self.hands_on_board-1, 0)

            if self.hands_on_board > 0:
                result["status"] = "HAND_ON_BOARD"
                result["board_poly"] = []
                results.append(result)
                continue

            img_cropped, [X, Y, W, H] = crop_board(image, board_poly)

            # HSV correction
            h, w, _ = img_cropped.shape
            # img_cropped_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
            # mean_value = np.mean(img_cropped_hsv[..., 2])
            # correction_factor = 200 / mean_value
            # img_cropped_hsv[..., 2] = np.uint8(np.clip(img_cropped_hsv[..., 2] * correction_factor, 0, 255))
            # img_cropped = cv2.cvtColor(img_cropped_hsv, cv2.COLOR_HSV2BGR)

            img_cropped_flipped = cv2.flip(img_cropped, 1)

            if self.detect_weights.endswith("onnx"): # ONNX
                boxes, labels, scores = self.detect_net(img_cropped)
            else: # Pytorch
                detections = self.detect_net(img_cropped, device = self.device, verbose=False, conf=CONF)[0]
                boxes = np.array([b.cpu().numpy() for b in detections.boxes.xyxyn]).reshape((-1,4))
                labels = [int(c.cpu()) for c in detections.boxes.cls]
                scores = [s.cpu() for s in detections.boxes.conf]

            # boxes_flipped, labels_flipped, scores_flipped = self.detect_net(img_cropped_flipped)

            # output = self.detect_net(
            #     [img_cropped, img_cropped_flipped],
            #     device = self.device,
            #     verbose=False,
            #     conf=CONF
            # )

            # detections = output[0]
            # detections_flipped = output[1]

            # boxes = np.array([b.cpu().numpy() for b in detections.boxes.xyxyn]).reshape((-1,4))
            # labels = [int(c.cpu()) + 1 for c in detections.boxes.cls]
            # scores = [s.cpu() for s in detections.boxes.conf]

            # boxes_flipped = np.array([b.cpu().numpy() for b in detections_flipped.boxes.xyxyn]).reshape((-1,4))
            # labels_flipped = [int(c.cpu()) + 1 for c in detections_flipped.boxes.cls]
            # scores_flipped = [s.cpu() for s in detections_flipped.boxes.conf]

            # boxes_flipped[:, [0, 2]]*=-1
            # boxes_flipped[:, [0, 2]]+=1
            # boxes_flipped[:, [2, 0]] = boxes_flipped[:, [0, 2]]

            # boxes = np.concatenate([boxes, boxes_flipped])
            # scores = np.concatenate([scores, scores_flipped])
            # labels = np.concatenate([labels, labels_flipped])

            for box in boxes:
                box[0] = (box[0] * W + X) / w
                box[2] = (box[2] * W + X) / w
                box[1] = (box[1] * H + Y) / h
                box[3] = (box[3] * H + Y) / h

            detections = (boxes, scores, labels)
            board, pieces = self.extract_position(image, detections, squares_corners)

            for p in pieces:
                if not p["piece"]:
                    continue
                p["box"][0] = (p["box"][0] * w - 0.5 * dw) / original.shape[1]
                p["box"][2] = (p["box"][2] * w - 0.5 * dw) / original.shape[1]
                p["box"][1] = (p["box"][1] * h - 0.5 * dh) / original.shape[0]
                p["box"][3] = (p["box"][3] * h - 0.5 * dh) / original.shape[0]

                p["center"][0] -= int(0.5 * dw)
                p["center"][1] -= int(0.5 * dh)

            for pt in result["board_poly"]:
                pt[0] -= int(0.5 * dw)
                pt[1] -= int(0.5 * dh)

            for sq in result["squares"]:
                for pt in sq:
                    pt[0] -= int(0.5 * dw)
                    pt[1] -= int(0.5 * dh)

            result["pieces"] = pieces
            result["board"] = board.fen().split(" ")[0]

            results.append(result)

            print(f"{time.time() - t} - Board extraction loop")
            t = time.time()

        return results

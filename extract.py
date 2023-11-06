import cv2
import numpy as np

from networks.detection.train import DetectNet
from networks.segmentation.train_segmentation import SegmentNet
from networks.classification.train_classif import ChessNet
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

# VIDEO_PATH = "/workspace/CL/data/test_images/carlsen.mp4"
# VIDEO_PATH = "/workspace/CL/data/test_images/nakamura.mp4"
VIDEO_PATH = "/workspace/CL/data/test_images/caruana.mp4"

class ChessboardParser():

    def __init__(self, device=-1, yolo_detect=False):
        self.class_names = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

        self.device_number = device
        if device < 0:
            self.device="cpu"
        else:
            self.device=f"cuda:{device}"

        self.yolo_detect=yolo_detect

        self.segment_net = YOLO('/workspace/CL/runs/segment/train3/weights/best.pt')
        # self.segment_net = SegmentNet(pretrained_path="/workspace/CL/model/segment/best_ref.torch")

        if self.yolo_detect:
            self.detect_net = YOLO('/workspace/CL/runs/detect/train15/weights/last.pt')
        else:
            self.detect_net = DetectNet(pretrained_path="/workspace/CL/model/detection/latest.torch", device=self.device)

        self.classif_net = ChessNet('/workspace/CL/model/classif/latest.torch', device_id=device, train=False)

        self.buffer = []
        self.buff_size = 30
        self.writer = None
        self.process_freq = 5
        self.save_individuals = True

        self.pieces_ratios = {}

        self.hands_net = mp.solutions.hands.Hands(max_num_hands=6, min_detection_confidence=0.1, static_image_mode=True)
        self.mask = None
        self.real_squares_corners = None
        self.pieces = None
        self.time_since_refresh=1000

        self.board = chess.Board()
        self.safe_board = chess.Board()
        self.hands_on_board = 0
        self.fen = ""
        self.unchanged_pos = 0
        self.needs_redraw=True
        self.board_img = None

        self.board_stability=3

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

        # DEEPLAB
        # self.mask = self.segment_net.infer(images = [image])[0]
        # self.mask = (self.mask * 255).astype(np.uint8)

        # results = self.segment_net(image, device=[0], verbose=False)[0]
        self.mask = mask

        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)

        if len(c) < 4:
            print("FOUND CONTOUR WITH LENGTH < 4")
            return False

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
                return False

        if len(approx) != 4:
            print("FAILED TO APPROXIMATE CONTOUR WITH 4 POINTS")
            return False

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

        self.real_squares_corners = []
        for r in range(8):
            for c in range(8):
                c1 = r * 9 + c
                c2 = c1 + 1
                c3 = c1 + 9
                c4 = c3 + 1

                self.real_squares_corners.append(np.array([real_squares[c] for c in [c1, c2, c4, c3]], np.int32))
        self.real_squares_corners = np.int32(self.real_squares_corners)


        board_poly = np.array([
            self.real_squares_corners[7][1],
            self.real_squares_corners[63][2],
            self.real_squares_corners[56][3],
            self.real_squares_corners[0][0]]
            )


        board_mask = cv2.fillPoly(np.zeros(self.mask.shape, np.uint8), [board_poly], color=255)
        board_mask = cv2.bitwise_xor(self.mask, board_mask)

        mask_overlay = np.sum(board_mask.reshape(-1)) / np.sum(self.mask.reshape(-1))

        self.real_squares_corners = np.float32(self.real_squares_corners)
        self.real_squares_corners[:,:,0] *= w / self.mask.shape[1]
        self.real_squares_corners[:,:,1] *= h / self.mask.shape[0]
        self.real_squares_corners = np.int32(self.real_squares_corners)

        self.mask = cv2.resize(self.mask, (w,h))

        if mask_overlay > 0.1:
            return False

        return True

    def extract_position(self, image, detections):

        boxes, scores, labels = detections

        h = image.shape[0]
        w = image.shape[1]

        real_square_centers = np.mean(self.real_squares_corners, axis=1, dtype=np.int32)
        self.pieces = [
            {
                "piece": None,
                "score": -1.0,
                "box": [],
                "center": real_square_centers[i],
            } for i in range(64)
        ]
        self.board.clear()

        squares_polygons = [Polygon(self.real_squares_corners[sq]) for sq in range(64)]

        # cropped_images = []
        # for box, score, label in zip(boxes, scores, labels):

        #     if score < 0.5:
        #         continue

        #     box_w = abs(box[2] - box[0])
        #     box_h = abs(box[3] - box[1])

        #     big_box = np.array(box).copy()
        #     margin = 0.1
        #     big_box[0] -= margin * box_w
        #     big_box[1] -= margin * box_h
        #     big_box[2] += margin * box_w
        #     big_box[3] += margin * box_h
        #     big_box = np.clip(big_box, 0.0, 1.0)
        #     big_box[0] *= image.shape[1]
        #     big_box[2] *= image.shape[1]
        #     big_box[1] *= image.shape[0]
        #     big_box[3] *= image.shape[0]
        #     big_box = np.int32(big_box)
        #     cropped_image = image[big_box[1]:big_box[3],big_box[0]:big_box[2]]
        #     cropped_images.append(cropped_image)

        # cv2.imwrite("/workspace/CL/test.jpg", cv2.hconcat([cv2.resize(img, (64,64)) for img in cropped_images]))

        # start = time.time()
        # labels = [self.class_names.index(label) + 1 for label in self.classif_net.infer(cropped_images)]
        # print(f"Pred time: {time.time() - start}")

        for box, score, label in zip(boxes, scores, labels):

            label -= 1

            if score < 0.5:
                continue

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
            # candidates = [i for i, c in enumerate(real_square_centers) if box_poly.contains(Point(c))]
            # candidates = range(64)
            # if len(candidates) == 0:
            #     continue
            # elif len(candidates) == 1:
            #     best_square = candidates[0]
            # else:
            #     best_square = max(candidates, key=lambda sq: squares_polygons[sq].intersection(box_poly).area)
            best_square, area = max(intersections, key=lambda x: x[1])


            if area > 0 and score > self.pieces[best_square]["score"]:
                self.pieces[best_square]["score"] = score
                self.pieces[best_square]["box"] = box

                piece = chess.Piece(piece_type = self.pieces_labels[(label)%6], color = label>=6)
                self.pieces[best_square]["piece"] = piece.symbol()
                self.board.set_piece_at(best_square, piece)
                # print(best_square, " - ", area, " - ", piece.symbol())














        # print("OUTPUT: ", r)
        # print("GROUND: ", gt)
        # print("---")

        # # Try to cluster empty squares into 2 groups to identify possible board rotations
        # empty_colors={}
        # for i in range(64):
        #     if not self.board.piece_at(i):
        #         mask = np.zeros((h,w,1), np.uint8)
        #         cv2.fillPoly(mask, [np.int32(self.real_squares_corners[i])], 255)
        #         mean = np.mean(cv2.mean(image, mask=mask))
        #         empty_colors[i] = mean

        # c, d = scipy.cluster.vq.kmeans([[v] for v in empty_colors.values()], 2)
        # white = max(c[0][0], c[1][0])
        # black = min(c[0][0], c[1][0])

        # empty_colors = {}
        # for k,v in empty_colors.items():
        #     empty_colors[k] = abs(v - white) < abs(v - black)

        # measured_colors = np.array(list(empty_colors.values()), np.int8)
        # target_colors = np.array([(int(k/8) + k%8) % 2 != 0 for k in empty_colors.keys()], np.int8)
        # loss = np.linalg.norm(measured_colors - target_colors, ord=1) / len(empty_colors)


        # # If squares colors don't match, rotate the board
        # if loss > 0.5:
        #     self.pieces = [self.pieces[int((7-i%8)*8+int(i/8))] for i in range(64)]

        # # Try to guess if board must be flipped

        # mean_W = np.mean([i for i, p in self.board.piece_map().items() if p.color])
        # mean_B = np.mean([i for i, p in self.board.piece_map().items() if not p.color])
        # if mean_W < mean_B:
        #     self.pieces = [self.pieces[int((7-int(i/8))*8+i%8)] for i in range(64)]


        # if not self.board.is_valid():
        #     status = self.board.status()

        #     # Replace bad kings by queens
        #     if status & chess.STATUS_TOO_MANY_KINGS:
        #         white_kings = [i for i in range(64) if self.pieces[i]["piece"]=="K"]
        #         black_kings = [i for i in range(64) if self.pieces[i]["piece"]=="k"]

        #         if len(white_kings) > 1:
        #             true_king = max(white_kings, key= lambda i: self.pieces[i]["score"])
        #             for i, p in enumerate(self.pieces):
        #                 if p["piece"] == "K" and i != true_king:
        #                     p["piece"] = "Q"
        #                     self.board.set_piece_at(i, chess.Piece(piece_type=chess.QUEEN, color=True))


        #         if len(black_kings) > 1:
        #             true_king = max(black_kings, key= lambda i: self.pieces[i]["score"])
        #             for i, p in enumerate(self.pieces):
        #                 if p["piece"] == "k" and i != true_king:
        #                     p["piece"] = "q"
        #                     self.board.set_piece_at(i, chess.Piece(piece_type=chess.QUEEN, color=False))

        #     white_bishops = [i for i in range(64) if self.pieces[i]["piece"]=="B"]
        #     if len(white_bishops) > 2:
        #         true_bishops = sorted(white_bishops, key= lambda i: self.pieces[i]["score"])[:2]
        #         for i, p in enumerate(self.pieces):
        #             if p["piece"] == "B" and i not in true_bishops:
        #                 p["piece"] = "N"
        #                 self.board.set_piece_at(i, chess.Piece(piece_type=chess.BISHOP, color=False))

        #     # TODO: Fix board based on status

        #     # missing_pieces = {}
        #     # for piece, max_nb in pieces_numbers.items():
        #     #     occ = len([p for p in self.pieces if p["piece"]==piece])
        #     #     missing_pieces[piece] = max(0, max_nb-occ)

        #     # for piece, max_nb in pieces_numbers.items():
        #     #     occ = len([p for p in self.pieces if p["piece"]==piece])
        #     #     if occ > max_nb:
        #     #         wrong_pieces = sorted([p for p in self.pieces if p["piece"] == piece], key=lambda x: -x["score"])[max_nb:]
        #     #         for wrong_piece in wrong_pieces:

        #     #             i = self.pieces.index(wrong_piece)
        #     #             old_piece = former_pieces[i]
        #     #             if missing_pieces.get(old_piece["piece"], 0) > 0:
        #     #                 print(f"Correction: {old_piece['piece']} was mistaken for {piece}")
        #     #                 wrong_piece["piece"] = old_piece["piece"]
        #     #                 missing_pieces[old_piece["piece"]] -= 1
        #     #             else:
        #     #                 for missing_piece in [p for p in missing_pieces if missing_pieces[p] > 0]:
        #     #                     if missing_piece.isupper() == piece.isupper() and missing_piece.lower() != "p":
        #     #                         print(f"Correction: {missing_piece} was mistaken for {piece}")
        #     #                         wrong_piece["piece"] = missing_piece
        #     #                         missing_pieces[missing_piece] -= 1


        return True

    def compute_accuracy(self, gt_pos):
        if not gt_pos:
            return [], 0.0
        pos = self.board.fen()
        different_squares = [(7 - int(i/8))*8 + i%8 for i in range(64) if pos[i] != gt_pos[i]]
        acc = (64 - len(different_squares)) / 64
        return different_squares, acc

    def draw_detections_on_image(self, image):

        h = image.shape[0]
        w = image.shape[1]


        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                box = self.pieces[sq]["box"]

                square_corners = np.float64(self.real_squares_corners[sq])
                square_corners[:,0] /= image.shape[1]
                square_corners[:,1] /= image.shape[0]

                top_center = np.mean([square_corners[1], square_corners[2]], axis = 0)
                bottom_center = np.mean([square_corners[0], square_corners[3]], axis = 0)
                left_center = np.mean([square_corners[0], square_corners[1]], axis = 0)
                right_center = np.mean([square_corners[2], square_corners[3]], axis = 0)

                square_w = abs((right_center -left_center)[0])
                square_h = abs((top_center - bottom_center)[1])

                box_w = abs(box[2] - box[0])
                box_h = abs(box[3] - box[1])
                box_h -= 0.5 * (box_w * square_h / square_w)

                angle = math.atan2(abs(top_center[0] - bottom_center[0]), abs(top_center[1] - bottom_center[1]))
                corrected_piece_height = box_h / math.cos(angle)
                ar = 100.0 * square_w / corrected_piece_height

                # print(ar)
                # print(square_corners)
                # print(top_center)
                # print(bottom_center)
                # print(square_w)
                # print(square_h)
                # print("--")

                if piece.symbol().lower() not in self.pieces_ratios:
                    self.pieces_ratios[piece.symbol().lower()] = []
                self.pieces_ratios[piece.symbol().lower()].append(ar)

                cv2.rectangle(image,
                    (int(box[0]*w), int(box[1]*h)),
                    (int(box[2]*w), int(box[3]*h)),
                    (255,255,255),
                    max(1, int(h / 500))
                )
                cv2.putText(
                    image,
                    f'{self.pieces[sq]["score"]:.2f}',
                    # f'{square_w:.2f} - {square_h:.2f}',
                    # piece.symbol(),
                    org = self.pieces[sq]["center"],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5 * h / 500,
                    color=(0,0,255),
                    thickness=max(1, int(h / 500))
                )

    def draw_board_on_image(self, image, highlights=[]):

        if self.needs_redraw:

            board_size = int(0.3 * image.shape[0])
            svg=chess.svg.board(
                self.safe_board,
                fill=dict.fromkeys(highlights, "#cc0000cc"),
                size=board_size
                )

            png = svg2png(bytestring=bytes(svg,'UTF-8'))
            pil_img = Image.open(BytesIO(png)).convert('RGB')
            self.board_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            self.needs_redraw = False

        h = image.shape[0]
        w = image.shape[1]
        bh = self.board_img.shape[0]
        bw = self.board_img.shape[1]
        image[h-bh:h,w-bw:w] = self.board_img

    def draw_squares_on_image(self, image):
        cv2.polylines(image, self.real_squares_corners, True, (255,0,0), max(1, int(image.shape[0] / 500)))

    def process_images(self, images):

        # Run segmentation and detection
        masks_batch = self.segment_net([cv2.resize(image, (480,480)) for image in images], device=self.device, verbose=False)

        for image_idx, image in enumerate(images):

            h, w, _ = image.shape

            if not masks_batch[image_idx].masks:
                continue

            mask = np.uint8(masks_batch[image_idx].masks.data.cpu().numpy() * 255)
            mask = np.swapaxes(mask, 0, 1)
            mask = np.swapaxes(mask, 1, 2)
            if mask.shape[2] > 1:
                mask = np.expand_dims(mask[:,:,0],2)

            squares_ret = self.extract_squares_coords(image, mask)
            mask = cv2.resize(mask, (w,h))

            if mask is not None:

                board_poly = np.array([
                    self.real_squares_corners[7][1],
                    self.real_squares_corners[63][2],
                    self.real_squares_corners[56][3],
                    self.real_squares_corners[0][0]]
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
                    cv2.putText(
                        image,
                        "HANDS ON BOARD",
                        org = (0,40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=h/500,
                        color=(0,255,0),
                        thickness=max(1, int(h/100))
                    )

                elif not squares_ret:
                    cv2.putText(
                        image,
                        "COULD NOT FIND BOARD",
                        org = (0,40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=h/500,
                        color=(0,255,0),
                        thickness=max(1, int(h/200))
                    )

                else:
                    [X, Y, W, H] = cv2.boundingRect(board_poly)
                    X = max(0, X)
                    Y = max(0, Y)
                    W = min(W, w - X - 1)
                    H = min(H, h - Y - 1)
                    target_ar = 540/960
                    ar = H/W
                    if ar < target_ar-0.01:
                        dh = target_ar * W - H
                        H += dh
                        Y -= 0.5 * dh
                    elif ar > target_ar+0.01:
                        dw = target_ar / H - W
                        W += dw
                        X -= 0.5 * dw
                    X = max(0, X)
                    Y = max(0, Y)
                    W = min(W, w - X - 1)
                    H = min(H, h - Y - 1)

                    img_cropped = image[int(Y):int(Y+H),int(X):int(X+W)]

                    if self.yolo_detect:
                        detections = self.detect_net([cv2.resize(img_cropped, (640,640))], device = [self.device_number], verbose=False)[0]
                        boxes = np.array([b.xyxy.cpu().numpy()[0] for b in detections.boxes], np.float32).reshape((-1,4)) / 640
                        labels = [int(b.cls.cpu().numpy()[0]) + 1 for b in detections.boxes]
                        scores = [b.conf.cpu().numpy()[0] for b in detections.boxes]
                    else:
                        boxes, scores, labels = self.detect_net([img_cropped], 0.5)[0]

                    for box in boxes:
                        # print(box)
                        box[0] = (box[0] * W + X) / w
                        box[2] = (box[2] * W + X) / w
                        box[1] = (box[1] * H + Y) / h
                        box[3] = (box[3] * H + Y) / h

                    if self.extract_position(image, (boxes, scores, labels)):

                        self.draw_detections_on_image(image)
                        if self.board.fen() == self.fen:
                            self.unchanged_pos += 1
                        else:
                            self.unchanged_pos = 0

                        if self.unchanged_pos > self.board_stability:
                            if self.board.fen() != self.safe_board.fen():
                                self.safe_board = self.board.copy()
                                self.needs_redraw = True
                                print("---------------")
                                print(self.safe_board)
                                print("---------------")

                        self.fen = self.board.fen()

                self.draw_board_on_image(image)
                self.draw_squares_on_image(image)

        return True

    def process_buffer(self):

        if len(self.buffer) == 0:
            return

        for i, (_, frame) in enumerate(self.buffer):

            # Adjust aspect ratio to fit train set
            target_ar = 540/960
            ar = frame.shape[0]/frame.shape[1]
            if ar < target_ar-0.01:
                border = int(0.5 * (target_ar * frame.shape[1] - frame.shape[0]))
                frame = cv2.copyMakeBorder(frame, border, border, 0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))
            elif ar > target_ar+0.01:
                border = int(0.5 * (frame.shape[0] / target_ar - frame.shape[1]))
                frame = cv2.copyMakeBorder(frame, 0,0, border, border, cv2.BORDER_CONSTANT, value = (0,0,0))


            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # frame = clahe.apply(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite("/workspace/CL/test.jpg", frame)

            # frame = np.uint8(np.float32(frame) * 0.7)
            # contrast_increase = 1.8
            # frame = cv2.convertScaleAbs(frame, alpha=contrast_increase, beta=127.0 * (1.0 -  contrast_increase))
            # frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

            # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # brightness = np.mean(hsv_frame[:,:,2])
            # print("Bright: ", brightness)

            # Sharpen image
            # strength = 1.0
            # blurred = cv2.GaussianBlur(frame, (15,15), 3)
            # frame = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
            # frame = np.uint8(frame * 0.5)
            # frame = cv2.flip(frame,1)

            self.buffer[i][1] = frame

        self.process_images([f for _, f in self.buffer])
        for frame_idx, image in self.buffer:
            if self.save_individuals:
                cv2.imwrite(f"/workspace/CL/output/{frame_idx}.jpg", image)
            else:
                if not writer:
                    vid_size = (1080,720)
                    fps = int(25/self.process_freq)
                    writer = cv2.VideoWriter("/workspace/CL/output/video.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, vid_size)
                writer.write(cv2.resize(image, vid_size))

        self.buffer.clear()

    def process_video(self, video_path):

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        # frame_idx = 2500
        start_idx = 2500
        end_index = start_idx + 2000

        frame_idx = start_idx
        while(ret):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
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

        # # print(json.dumps(self.pieces_ratios, indent=4))
        # for symbol in ["k", "q", "b", "n", "r", "p"]:
        #     ratios = self.pieces_ratios[symbol]
        #     min_ratio = min(ratios)
        #     max_ratio = max(ratios)
        #     mean_ratio = np.mean(ratios)
        #     std_ratio = np.std(ratios)
        #     # print(f"{symbol}:  {min_ratio:.1f} - {max_ratio:.1f} - {mean_ratio:.1f} - {std_ratio:.2f}")

        # all_ratios = []
        # for ratios in self.pieces_ratios.values():
        #     all_ratios.extend(ratios)
        # # print(np.array(all_ratios).shape)
        # centers, d = scipy.cluster.vq.kmeans(all_ratios, k_or_guess=6)
        # # print(centers)

        # regrouped = [[] for _ in centers]
        # for symbol in ["k", "q", "b", "n", "r", "p"]:
        #     for ratio in self.pieces_ratios[symbol]:
        #         centroid_idx = np.argmin([abs(ratio - c) for c in centers])
        #         regrouped[centroid_idx].append(symbol)
        # # for group in regrouped:
        #     # print(group)

        cap.release()
        if self.writer:
            self.writer.release()



if __name__ == "__main__":
    with(open("/workspace/CL/data/gt.json"))as f:
        gt = json.loads(f.read())["positions"]

    parser = ChessboardParser(device=0, yolo_detect=True)
    parser.process_video(VIDEO_PATH)



    # image_path = "/workspace/CL/data/test_images/sequence2/Screenshot from 2023-08-25 02-00-46.png"
    # # image_path = "/workspace/CL/data/test_images/1000011471.jpg"
    # # image_path = "/workspace/CL/data/test_images/1000011474.jpg"

    # # images = sorted(glob.glob("/workspace/CL/data/test_images/sequence3/*.png"))
    # images = [image_path]

    # boards = []
    # for k, path in enumerate(images):
    #     image_name = Path(path).name
    #     image = cv2.imread(image_path)

    #     ret, pieces = process_image(image)
    #     if not ret:
    #         print(f"Failed to process {image_path}")
    #         continue

    #     position = position_from_pieces(pieces)
    #     diff, acc = compute_accuracy(pieces, gt.get(image_name))
    #     board = draw_board_on_image(image, pieces, diff)

    #     boards.append(board)
    #     cv2.imwrite(f"/workspace/CL/output/{image_name}", image)
    #     print(f"{position} (acc: {int(acc*100)}%)")

    # boards = cv2.hconcat(boards)
    # cv2.imwrite("/workspace/CL/boards.jpg", boards)
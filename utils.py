import glob
import os
import shutil
import cv2
import numpy as np
import json
from pathlib import Path
import chess.svg
from cairosvg import svg2png
from PIL import Image
from io import BytesIO

from tqdm import tqdm

import chess

import uuid

def make_square_image(img):
    ar = img.shape[0]/img.shape[1]
    borders = np.array([0,0,0,0])
    if ar < 1.00:
        borders[[0,1]] = int(0.5 * (img.shape[1] - img.shape[0]))
    elif ar > 1.00:
        borders[[2,3]] = int(0.5 * (img.shape[0] - img.shape[1]))

    output = cv2.copyMakeBorder(img, borders[0], borders[1], borders[2], borders[3], cv2.BORDER_CONSTANT, value = (0,0,0))
    return output, borders

def crop_board(img, corners):

    h, w, _ = img.shape

    [X, Y, W, H] = cv2.boundingRect(corners)
    center = (X + 0.5 * W, Y + 0.5 * H)

    W *= 1.2
    H *= 1.2

    ar = H/W
    if ar < 1.0:
        H = W
    else:
        W = H

    X = center[0] - 0.5 * W
    Y = center[1] - 0.5 * H

    X = int(X)
    Y = int(Y)
    W = int(W)
    H = int(H)

    left = max(0, -X)
    top = max(0, -Y)
    right = max(0, X + left + W - w)
    bottom = max(0, Y + top + H - h)

    if np.max([left, top, right, bottom]) > 0:
        output = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (0,0,0))
    else:
        output = img.copy()

    return output[Y+top:Y+top+H,X+left:X+left+W], [X, Y, W, H], [left, top, right, bottom]

def align_image(img, board_pos, output_size=640):
    board_size = 0.8
    dst_pts = []
    for c in range(9):
        for r in range(9):
            x = 0.5 + (c-4) * board_size / 16
            y = 0.5 + (r-4) * board_size / 16
            dst_pts.append([x * output_size, y * output_size])
    dst_pts = np.int32(dst_pts)

    src_pts = np.array(board_pos)
    src_pts[:,1] *= -1
    src_pts[:,1] += 1
    src_pts[:,0] *= img.shape[1]
    src_pts[:,1] *= img.shape[0]

    for i, pt in enumerate(src_pts):
        cv2.putText(
            img,
            # f'{pieces[i]["score"]:.2f}',
            f"{i}",
            org = np.int32(pt),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.2 * img.shape[0] / 500,
            color=(255,255,255),
            thickness=max(1, int(img.shape[0] / 800))
        )

    H = cv2.findHomography(src_pts, dst_pts)
    mat = H[0]

    output = cv2.warpPerspective(img, mat, (output_size, output_size))
    newPts = cv2.perspectiveTransform(np.array([src_pts]), mat)
    print(newPts[0][41])

    for i, dst_pt in enumerate(newPts[0]):
        cv2.putText(
            output,
            # f'{pieces[i]["score"]:.2f}',
            f"{i}",
            org = np.int32(dst_pt),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.2 * img.shape[0] / 500,
            color=(0,0,255),
            thickness=max(1, int(img.shape[0] / 800))
        )

    for i, dst_pt in enumerate(dst_pts):
        cv2.putText(
            output,
            # f'{pieces[i]["score"]:.2f}',
            f"{i}",
            org = dst_pt,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.2 * img.shape[0] / 500,
            color=(0,255,0),
            thickness=max(1, int(img.shape[0] / 800))
        )

    return output

def draw_detections_on_image(image, pieces):

    h = image.shape[0]
    w = image.shape[1]

    for piece in pieces:

        if not piece["piece"]:
            continue
        box = piece["box"]

        cv2.rectangle(image,
            (int(box[0]*w), int(box[1]*h)),
            (int(box[2]*w), int(box[3]*h)),
            (255,255,255),
            max(1, int(h / 500))
        )
        cv2.putText(
            image,
            f'{piece["piece"]} - {piece["score"]:.2f}',
            # piece.symbol(),
            org = piece["center"],
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5 * h / 1000,
            color=(0,0,255),
            thickness=max(1, int(h / 500))
        )

def draw_board_on_image(board, image, highlights=[]):

    board_size = int(0.3 * image.shape[0])
    svg=chess.svg.board(
        board,
        fill=dict.fromkeys(highlights, "#cc0000cc"),
        size=board_size
        )

    png = svg2png(bytestring=bytes(svg,'UTF-8'))
    pil_img = Image.open(BytesIO(png)).convert('RGB')
    board_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h = image.shape[0]
    w = image.shape[1]
    bh = board_img.shape[0]
    bw = board_img.shape[1]
    image[h-bh:h,w-bw:w] = board_img

def draw_squares_on_image(image, squares_corners):
    cv2.polylines(image, squares_corners, True, (255,0,0), max(1, int(image.shape[0] / 500)))

def draw_results_on_image(image, results):

    if results["info"] == "ok":

        draw_squares_on_image(image, results["squares"])
        draw_detections_on_image(image, results["pieces"])

        board = chess.Board.empty()
        board.set_board_fen(results["board"])
        draw_board_on_image(board, image)

    else:
        h = image.shape[0]
        cv2.putText(
            image,
            results["info"],
            org = (0,40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=h/500,
            color=(0,255,0),
            thickness=max(1, int(h/200))
        )

def yolo_box_intersect(box1, box2):
    start_x = max(box1[0], box2[0])
    end_x = max(start_x, min(box1[2], box2[2]))

    start_y = max(box1[1], box2[1])
    end_y = max(start_y, min(box1[3], box2[3]))

    return [start_x, start_y, end_x, end_y]


def yolo_boxes_iou(box1, box2):
    x1, y1, x2, y2 = yolo_box_intersect(box1, box2)
    inter = (x2 - x1) * (y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter/union


# merge(["/workspace/CL/dataset_augment", "/workspace/CL/dataset3"], "/workspace/CL/dataset_merge")
# gen_yolo_annots_seg()
# gen_yolo_annots()
# split()
# prepare_dataset("/workspace/CL/data/dataset5", "/workspace/CL/data/dataset5_preprocessed")
# gen_masks()
# gen_pieces_dataset()
# extract_kings_queens()

# files = glob.glob("/workspace/ChessLink/data/dataset_test_CL7/*.jpg")
# visualize_annots(files[2])
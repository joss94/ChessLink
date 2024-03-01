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
    if ar < 0.99:
        borders[[0,1]] = int(0.5 * (img.shape[1] - img.shape[0]))
    elif ar > 1.01:
        borders[[2,3]] = int(0.5 * (img.shape[0] / img.shape[1]))
    borders += int(0.2 * img.shape[0])
    output = cv2.copyMakeBorder(img, borders[0], borders[1], borders[2], borders[3], cv2.BORDER_CONSTANT, value = (0,0,0))
    return output

def crop_board(img, corners):

    h, w, _ = img.shape
    [X, Y, W, H] = cv2.boundingRect(corners)

    X -= 0.1 * W
    Y -= 0.1 * H
    W *= 1.2
    H *= 1.2
    # X = max(0, X)
    # Y = max(0, Y)
    # W = min(W, w - X - 1)
    # H = min(H, h - Y - 1)
    ar = H/W
    if ar < 0.99:
        dh = W - H
        H += dh
        Y = max(0, Y - 0.5 * dh)
    elif ar > 1.01:
        dw = H - W
        W += dw
        X =  max(0, X - 0.5 * dw)

    return img[int(Y):int(Y) + int(H),int(X):int(X)+int(W)], [int(X), int(Y), int(W), int(H)]

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

    if results["status"] == "OK":

        draw_squares_on_image(image, results["squares"])
        draw_detections_on_image(image, results["pieces"])

        board = chess.Board.empty()
        board.set_board_fen(results["board"])
        draw_board_on_image(board, image)

    else:
        h = image.shape[0]
        cv2.putText(
            image,
            results["status"],
            org = (0,40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=h/500,
            color=(0,255,0),
            thickness=max(1, int(h/200))
        )


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
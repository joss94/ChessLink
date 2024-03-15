import glob
import os
import shutil
import cv2
import numpy as np
import json
from pathlib import Path
import random
import re

from multiprocessing import Pool

from tqdm import tqdm

import uuid

from utils.image_utils import make_square_image, crop_board

import hashlib


def clean(dataset_path):

    path = dataset_path + "/*.jpg"
    image_file_paths = glob.glob(path, recursive=True)

    # Remove images with no GT
    images_with_no_GT = []
    for path in image_file_paths:
        json_path = path.replace(".jpg", ".json")
        if not os.path.exists(json_path):
            images_with_no_GT.append(path)

    print(
        f"Will remove {len(images_with_no_GT)}/{len(image_file_paths)} images with no GT"
    )
    for path in images_with_no_GT:
        os.remove(path)


def merge(datasets, dataset_dst):
    os.makedirs(dataset_dst, exist_ok=True)

    image_file_paths = []
    for dataset in datasets:
        dataset_images = glob.glob(dataset + "/*.jpg", recursive=True)
        print(len(dataset_images))
        image_file_paths.extend(dataset_images)

    for image_path in tqdm(image_file_paths):
        data_name = Path(image_path).stem
        shutil.copy(image_path, os.path.join(dataset_dst, f"{data_name}.jpg"))
        shutil.copy(
            image_path.replace(".jpg", ".json"),
            os.path.join(dataset_dst, f"{data_name}.json"),
        )


def merge_yolo(datasets, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir) / "train", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "labels", exist_ok=True)

    image_file_paths = []
    for dataset in datasets:

        images = glob.glob(dataset + "/train/images/*.jpg")
        if "yolo_merge" in dataset:
            images = images[:5000]

        for image_path in tqdm(images):
            label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")
            data_name = Path(image_path).stem
            dst_img_path = Path(dst_dir) / "train" / "images" / f"{data_name}.jpg"
            dst_label_path = Path(dst_dir) / "train" / "labels" / f"{data_name}.txt"
            if not dst_img_path.exists():
                shutil.copy(image_path, str(dst_img_path))
                shutil.copy(label_path, str(dst_label_path))

    with open(Path(dst_dir) / "data.yaml", "w+") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../../chessred_test_yolo/images\n")
        f.write("test: ../../chessred_test_yolo/images\n")
        f.write(f"nc: 12\n")
        f.write(
            f"names: ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']\n"
        )

        f.write(f"augment: True\n")


def visualize_annots(img_path):

    img = cv2.imread(img_path)
    H = img.shape[0]
    W = img.shape[1]

    with open(img_path.replace(".jpg", ".json")) as f:
        annots = json.loads(f.read())

    for piece in annots["pieces"]:
        box = np.array(piece["bbox"])
        label = piece["piece"]
        cv2.rectangle(
            img,
            (int(box[0] * W), int(box[1] * H)),
            (int(box[2] * W), int(box[3] * H)),
            (255, 255, 255),
            max(1, int(img.shape[0] / 500)),
        )
        cv2.putText(
            img,
            # f'{pieces[i]["score"]:.2f}',
            label,
            org=(int(box[0] * W), int(box[1] * H)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4 * img.shape[0] / 500,
            color=(0, 255, 0),
            thickness=max(1, int(img.shape[0] / 500)),
        )
    cv2.imwrite("/workspace/ChessLink/visu.jpg", img)


def visualize_annots_yolo(dataset_path):

    files = glob.glob(f"{dataset_path}/*/*/*.jpg")
    # files = [os.path.join(dataset_path, "train/images/data_14c114f2-907d-11ee-9884-a036bc2aad3a.jpg")]

    for file in files:
        img_path = Path(file)

        annots_path = img_path.parent.parent / "labels" / (str(img_path.stem) + ".txt")

        img = cv2.imread(str(img_path))

        H = img.shape[0]
        W = img.shape[1]

        with open(annots_path) as f:
            annots = f.readlines()

        for piece in annots:
            elems = piece.split(" ")
            label = elems[0]
            c_x = float(elems[1])
            c_y = float(elems[2])
            w = float(elems[3])
            h = float(elems[4])

            cv2.rectangle(
                img,
                (int((c_x - 0.5 * w) * W), int((c_y - 0.5 * h) * H)),
                (int((c_x + 0.5 * w) * W), int((c_y + 0.5 * h) * H)),
                (255, 255, 255),
                max(1, int(img.shape[0] / 500)),
            )
            cv2.putText(
                img,
                label,
                org=(int(c_x * W), int(c_y * H)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4 * img.shape[0] / 500,
                color=(0, 255, 0),
                thickness=max(1, int(img.shape[0] / 500)),
            )
        cv2.imwrite("/workspace/ChessLink/visu.jpg", img)
        a = input()


def gen_yolo_annot(
    path, dst_dir, train_split, val_split, class_names, diff_color, overwrite
):

    annots_path = Path(dst_dir) / "train" / "labels" / (str(Path(path).stem) + ".txt")
    img_path = Path(dst_dir) / "train" / "images" / (str(Path(path).stem) + ".jpg")

    # Image was already parsed, don't do it again
    if not overwrite and (img_path.exists() and annots_path.exists()):
        return 0

    img = cv2.imread(path)
    h, w, _ = img.shape

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    center_sz = 0.1
    img_hsv_center = img_hsv[
        int((0.5 - center_sz) * h) : int((0.5 + center_sz) * h),
        int((0.5 - center_sz) * w) : int((0.5 + center_sz) * w),
    ]
    mean_value = np.mean(img_hsv[..., 2])
    correction_factor = 200 / mean_value
    img_hsv[..., 2] = np.uint8(np.clip(img_hsv[..., 2] * correction_factor, 0, 255))
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    with open(path.replace(".jpg", ".json")) as f:
        annots = json.loads(f.read())

    board_poly = np.array(annots["board"])
    board_poly[:, 0] *= w
    board_poly[:, 1] *= -h
    board_poly[:, 1] += h
    board_poly = np.int32(board_poly)

    img_cropped, [X, Y, W, H] = crop_board(img, board_poly)

    annots_txt = ""

    for piece in annots["pieces"]:
        box = piece["bbox"]

        box[0] = (box[0] * w - X) / W
        box[2] = (box[2] * w - X) / W
        box[1] = (box[1] * h - Y) / H
        box[3] = (box[3] * h - Y) / H

        box = np.clip(box, 0.0, 1.0)

        if box[0] >= 1.0 or box[2] <= 0 or box[1] >= 1.00 or box[3] <= 0:
            continue

        label = piece["piece"]
        if not diff_color:
            label = label.lower()
        p_class = class_names.index(label)
        center_x = 0.5 * (box[0] + box[2])
        center_y = 0.5 * (box[1] + box[3])
        width = abs(box[2] - box[0])
        height = abs(box[1] - box[3])
        annots_txt += f"{p_class} {center_x} {center_y} {width} {height}\n"

    # shutil.copyfile(path, str(img_path))
    cv2.imwrite(str(img_path), img_cropped)

    with open(annots_path, "w+") as f:
        f.write(annots_txt)

    return 0


def convert_roboflow(src_directory="", dst_directory=""):

    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory, exist_ok=True)

    with open(os.path.join(src_directory, "data.yaml"), "r") as f:
        data_str = f.read()

    with open(os.path.join(dst_directory, "data.yaml"), "w") as f:
        data_str = re.sub(
            r"names: .*",
            "names: ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']",
            data_str,
        )
        data_str = re.sub(r"nc: .*", "nc: 12", data_str)
        f.write(data_str)

    for subset in ["train", "valid", "test"]:

        print(f"Processing {subset}")

        if not os.path.exists(os.path.join(dst_directory, subset)):
            os.makedirs(os.path.join(dst_directory, subset), exist_ok=True)
            os.makedirs(os.path.join(dst_directory, subset, "images"), exist_ok=True)
            os.makedirs(os.path.join(dst_directory, subset, "labels"), exist_ok=True)

        images = glob.glob(os.path.join(src_directory, subset) + "/images/*.jpg")
        for image_path in tqdm(images):

            img = cv2.imread(image_path)
            h, w, c = img.shape

            image_name = Path(image_path).stem
            with open(
                os.path.join(src_directory, subset, "labels", f"{image_name}.txt"), "r"
            ) as f:
                label_lines = f.readlines()

            board = []
            for line in label_lines:
                label = int(line.split(" ")[0])
                if label == 12:
                    board = [float(s) for s in line.split(" ")[1:]]

                elif label == 13:
                    coords = [float(s) for s in line.split(" ")[1:]]
                    img[
                        int((coords[1] - 0.5 * coords[3]) * h) : int(
                            (coords[1] + 0.5 * coords[3]) * h
                        ),
                        int((coords[0] - 0.5 * coords[2]) * w) : int(
                            (coords[0] + 0.5 * coords[2]) * w
                        ),
                    ] = 0

            # Board position must be annotated
            if not len(board):
                continue

            # assert(False)

            board_poly = np.array(
                [
                    [board[0] - 0.5 * board[2], board[1] - 0.5 * board[3]],
                    [board[0] - 0.5 * board[2], board[1] + 0.5 * board[3]],
                    [board[0] + 0.5 * board[2], board[1] - 0.5 * board[3]],
                    [board[0] + 0.5 * board[2], board[1] + 0.5 * board[3]],
                ]
            )

            board_poly[:, 0] *= w
            board_poly[:, 1] *= h
            board_poly = np.int32(board_poly)

            img_cropped, [X, Y, W, H] = crop_board(img, board_poly)

            annots_txt = ""
            for i, line in enumerate(label_lines):
                label = int(line.split(" ")[0])
                if label >= 12:
                    continue

                coords = [float(s) for s in line.split(" ")[1:]]
                box = np.array(
                    [
                        coords[0] - 0.5 * coords[2],
                        coords[1] - 0.5 * coords[3],
                        coords[0] + 0.5 * coords[2],
                        coords[1] + 0.5 * coords[3],
                    ]
                )

                box[0] = max(0.0, (box[0] * w - X) / W)
                box[2] = min(1.0, (box[2] * w - X) / W)
                box[1] = max(0.0, (box[1] * h - Y) / H)
                box[3] = min(1.0, (box[3] * h - Y) / H)

                box = np.clip(box, 0.0, 1.0)

                if box[2] <= box[0] or box[3] <= box[1]:
                    continue

                center_x = 0.5 * (box[0] + box[2])
                center_y = 0.5 * (box[1] + box[3])
                width = abs(box[2] - box[0])
                height = abs(box[1] - box[3])

                annots_txt += f"{label} {center_x} {center_y} {width} {height}\n"

            with open(
                os.path.join(dst_directory, subset, "labels", f"{image_name}.txt"), "w"
            ) as f:
                f.write(annots_txt)

            cv2.imwrite(
                os.path.join(dst_directory, subset, "images", f"{image_name}.jpg"),
                img_cropped,
            )

    # Create valid dir same as train dir if not exists
    train_dir = os.path.join(dst_directory, "train")
    valid_dir = os.path.join(dst_directory, "valid")
    if not os.path.exists(valid_dir):
        shutil.copytree(
            train_dir,
            valid_dir,
            symlinks=False,
            ignore=None,
            copy_function=shutil.copy2,
            ignore_dangling_symlinks=False,
            dirs_exist_ok=False,
        )


def gen_yolo_annots(
    data_directory="/workspace/ChessLink/data/dataset_test_CL21",
    dst_dir="/workspace/ChessLink/data/dataset_yolo_21",
    diff_color=True,
    overwrite=False,
):

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir) / "train", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "labels", exist_ok=True)

    path = data_directory + "/*.jpg"
    image_file_paths = glob.glob(path, recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]
    # image_file_paths = image_file_paths[:100]

    train_split = 0.9
    val_split = 1.0 - train_split

    if diff_color:
        class_names = [
            "B",
            "K",
            "N",
            "P",
            "Q",
            "R",
            "board",
            "mask",
            "b",
            "k",
            "n",
            "p",
            "q",
            "r",
        ]
    else:
        class_names = ["p", "n", "b", "r", "q", "k"]

    params = [
        (
            image_file_paths[i],
            dst_dir,
            train_split,
            val_split,
            class_names,
            diff_color,
            overwrite,
        )
        for i in range(len(image_file_paths))
    ]

    for p in tqdm(params):
        gen_yolo_annot(*p)
    # with Pool(4) as p:
    #     print("Starting...")
    #     # p.starmap(gen_yolo_annot, params)
    #     for _ in tqdm(p.starmap(gen_yolo_annot, params)):
    #         pass

    # for path in tqdm(image_file_paths):

    with open(Path(dst_dir) / "data.yaml", "w+") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../../chessred_test_yolo/images\n")
        f.write("test: ../../chessred_test_yolo/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(
            f"names: ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']\n"
        )

        f.write(f"augment: True\n")
        # f.write(f"mosaic: 0.0\n")


def change_yolo_split(dir, train=0.8, val=0.15, test=0.05):
    images = glob.glob(dir + "/*/images/*.jpg")

    for image_path in images:
        label_path = image_path.replace("images", "labels").replace("jpg", "txt")

        a = np.random()
        if a > test:
            dst_image_path = image_path.replace("train", "test").replace(
                "valid", "test"
            )
            dst_label_path = dst_image_path.replace("images", "labels").replace(
                "jpg", "txt"
            )
            shutil.copyfile(image_path, dst_image_path)
        elif a > val:
            dst_image_path = image_path.replace("train", "valid").replace(
                "test", "valid"
            )
            dst_label_path = dst_image_path.replace("images", "labels").replace(
                "jpg", "txt"
            )
            shutil.copyfile(image_path, dst_image_path)
        else:
            dst_image_path = image_path.replace("valid", "train").replace(
                "test", "train"
            )
            dst_label_path = dst_image_path.replace("images", "labels").replace(
                "jpg", "txt"
            )
            shutil.copyfile(image_path, dst_image_path)


def extract_kings_queens(
    data_directory="/workspace/ChessLink/data/dataset_test_CL5",
    dst_dir="/workspace/ChessLink/data/dataset_qk",
):

    os.makedirs(dst_dir, exist_ok=True)

    path = data_directory + "/*.jpg"
    image_file_paths = glob.glob(path, recursive=True)

    i = 0
    for path in tqdm(image_file_paths):

        with open(path.replace(".jpg", ".json")) as f:
            annots = json.loads(f.read())

        image = cv2.imread(path)

        H = image.shape[0]
        W = image.shape[1]

        for piece in annots["pieces"]:
            box = piece["bbox"]
            box = np.clip(box, 0.0, 1.0)

            if box[0] >= W or box[2] <= 0 or box[1] >= H or box[3] <= 0:
                continue

            if piece["piece"].lower() not in ["q", "k"]:
                continue

            # print(box)
            piece_img = image[box[1] : box[3], box[0] : box[2]]

            name = (
                f"queen_{i}.jpg" if piece["piece"].lower() == "q" else f"king_{i}.jpg"
            )

            cv2.imwrite(str(Path(dst_dir) / name), piece_img)
            i += 1


def gen_yolo_annots_seg(
    data_directory="/workspace/ChessLink/data/dataset_test_CL23",
    dst_dir="/workspace/ChessLink/data/dataset_yolo_seg_3",
):

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir) / "train", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "labels", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid" / "labels", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test" / "labels", exist_ok=True)

    path = data_directory + "/*.jpg"
    image_file_paths = glob.glob(path, recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]
    # image_file_paths = image_file_paths[:1000]

    train_split = 0.8
    val_split = 0.2

    for path in tqdm(image_file_paths):

        rand = np.random.uniform(0, 1)

        if rand < train_split:
            annots_path = (
                Path(dst_dir) / "train" / "labels" / (str(Path(path).stem) + ".txt")
            )
            img_path = Path(dst_dir) / "train" / "images" / Path(path).name
        elif rand < train_split + val_split:
            annots_path = (
                Path(dst_dir) / "valid" / "labels" / (str(Path(path).stem) + ".txt")
            )
            img_path = Path(dst_dir) / "valid" / "images" / Path(path).name
        else:
            annots_path = (
                Path(dst_dir) / "test" / "labels" / (str(Path(path).stem) + ".txt")
            )
            img_path = Path(dst_dir) / "test" / "images" / Path(path).name

        with open(path.replace(".jpg", ".json")) as f:
            annots = json.loads(f.read())

        with open(annots_path, "w+") as f:

            annot = "0"

            board_corners = [
                annots["board"][0],
                annots["board"][8],
                annots["board"][80],
                annots["board"][72],
            ]

            for corner in board_corners:
                annot += f" {corner[0]} {1.0-corner[1]}"
            f.write(f"{annot}\n")

        cv2.imwrite(str(img_path), cv2.resize(cv2.imread(path), (640, 640)))

        with open(Path(dst_dir) / "data.yaml", "w+") as f:
            f.write("train: ../train/images\n")
            f.write("val: ../valid/images\n")
            f.write("test: ../test/images\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['board']\n")
            f.write(f"augment: True\n")


def gen_yolo_annots_seg_with_pieces(
    data_directory="/workspace/ChessLink/data/dataset_test_CL15",
    dst_dir="/workspace/ChessLink/data/dataset_yolo_seg_3",
    diff_color=True,
    diff_pieces=True,
    use_board=True,
    overwrite=False,
):

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir) / "train", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "train" / "labels", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid" / "labels", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test" / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test" / "labels", exist_ok=True)

    path = data_directory + "/*.jpg"
    image_file_paths = glob.glob(path, recursive=True)
    # image_file_paths = image_file_paths[:1000]

    train_split = 0.8
    val_split = 0.2

    if diff_color:
        if diff_pieces:
            class_names = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]
        else:
            class_names = ["black", "white"]
    else:
        if diff_pieces:
            class_names = ["p", "n", "b", "r", "q", "k"]
        else:
            class_names = ["piece"]

    if use_board:
        class_names.append("board")

    for path in tqdm(image_file_paths):

        np.random.seed(abs(hash(path)) % (10 ** 8))
        rand = np.random.uniform(0, 1)

        if rand < train_split:
            annots_path = (
                Path(dst_dir) / "train" / "labels" / (str(Path(path).stem) + ".txt")
            )
            img_path = Path(dst_dir) / "train" / "images" / Path(path).name
        elif rand < train_split + val_split:
            annots_path = (
                Path(dst_dir) / "valid" / "labels" / (str(Path(path).stem) + ".txt")
            )
            img_path = Path(dst_dir) / "valid" / "images" / Path(path).name
        else:
            annots_path = (
                Path(dst_dir) / "test" / "labels" / (str(Path(path).stem) + ".txt")
            )
            img_path = Path(dst_dir) / "test" / "images" / Path(path).name

        if not overwrite and (img_path.exists() and annots_path.exists()):
            continue

        img = cv2.imread(path)
        h, w, _ = img.shape

        with open(path.replace(".jpg", ".json")) as f:
            annots = json.loads(f.read())

        board_poly = np.array(annots["board"])
        board_poly[:, 0] *= w
        board_poly[:, 1] *= -1
        board_poly[:, 1] += 1
        board_poly[:, 1] *= h
        board_poly = np.int32(board_poly)

        img_cropped, [X, Y, W, H] = crop_board(img, board_poly)

        img = cv2.imread(path)
        img_size = img.shape[0]

        with open(annots_path, "w+") as f:

            n_pieces = len(annots["pieces"])
            mask = cv2.imread(path.replace(".jpg", "_mask0024.png"))
            values = sorted(set(mask.reshape((-1))))

            if len(values) != n_pieces + 2:
                print(f"Invalid image: {path}")
                continue

            for piece in annots["pieces"]:
                if diff_pieces:
                    if diff_color:
                        label = piece["piece"]
                    else:
                        label = piece["piece"].lower()
                else:
                    if diff_color:
                        label = "black" if piece["piece"].islower() else "white"
                    else:
                        label = "piece"
                p_class = class_names.index(label)

                piece_mask = np.uint8((mask == values[piece["index"]])[:, :, 0])
                contours, hierarchy = cv2.findContours(
                    piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                annot = str(p_class)
                for pt in contours[0]:
                    x = (pt[0][0] - X) / W
                    y = (pt[0][1] - Y) / H
                    annot += f" {x} {y}"
                f.write(f"{annot}\n")

            if use_board:
                annot = f"{class_names.index('board')}"

                board_corners = [
                    annots["board"][0],
                    annots["board"][8],
                    annots["board"][80],
                    annots["board"][72],
                ]

                for corner in board_corners:
                    x = (corner[0] * w - X) / W
                    y = ((1.0 - corner[1]) * h - Y) / H
                    annot += f" {x} {y}"
                f.write(f"{annot}\n")

        cv2.imwrite(str(img_path), img_cropped)

        with open(Path(dst_dir) / "data.yaml", "w+") as f:
            f.write("train: ../train/images\n")
            f.write("val: ../valid/images\n")
            f.write("test: ../test/images\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")


def chessred_2_yolo(
    src_dir="/workspace/ChessLink/data/chessred_test",
    dst_dir="/workspace/ChessLink/data/chessred_test_yolo2",
):

    class_indices = [6, 9, 7, 8, 10, 11, 0, 3, 1, 2, 4, 5, 12]

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir) / "images", exist_ok=True)
    os.makedirs(Path(dst_dir) / "labels", exist_ok=True)

    path = src_dir + "/*/*.jpg"
    image_file_paths = glob.glob(path, recursive=True)

    with open("./data/chessred_test/annotations.json") as f:
        annots = json.load(f)

    for path in tqdm(image_file_paths):

        annots_path = Path(dst_dir) / "labels" / (str(Path(path).stem) + ".txt")
        img_path = Path(dst_dir) / "images" / (str(Path(path).stem) + ".jpg")

        annot = next(a for a in annots["images"] if a["file_name"] == Path(path).name)
        img_id = annot["id"]
        corners_annot = next(
            c for c in annots["annotations"]["corners"] if c["image_id"] == img_id
        )
        corners = np.int32(
            [
                corners_annot["corners"]["top_left"],
                corners_annot["corners"]["top_right"],
                corners_annot["corners"]["bottom_left"],
                corners_annot["corners"]["bottom_right"],
            ]
        )

        img = cv2.imread(path)
        h, w, _ = img.shape

        img_cropped, [X, Y, W, H] = crop_board(img, corners)

        pieces = [p for p in annots["annotations"]["pieces"] if p["image_id"] == img_id]

        annots_txt = ""

        for piece in pieces:
            box = piece["bbox"]

            box[0] = (box[0] - X) / W
            box[2] /= W
            box[1] = (box[1] - Y) / H
            box[3] /= H

            p_class = class_indices[piece["category_id"]]
            p_class = p_class % 6
            center_x = box[0] + 0.5 * box[2]
            center_y = box[1] + 0.5 * box[3]
            width = box[2]
            height = box[3]
            annots_txt += f"{p_class} {center_x} {center_y} {width} {height}\n"

        # shutil.copyfile(path, str(img_path))
        cv2.imwrite(str(img_path), img_cropped)

        with open(annots_path, "w+") as f:
            f.write(annots_txt)


def gen_pieces_dataset_from_yolo(
    src_dir="/workspace/ChessLink/data/dataset_test_CL23",
    dst_dir="/workspace/ChessLink/data/dataset_pieces",
    target_size=(64, 64),
):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir) / "train", exist_ok=True)
    os.makedirs(Path(dst_dir) / "valid", exist_ok=True)
    os.makedirs(Path(dst_dir) / "test", exist_ok=True)

    class_names = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]
    for label in class_names:
        os.makedirs(Path(dst_dir) / "train" / label, exist_ok=True)
        os.makedirs(Path(dst_dir) / "val" / label, exist_ok=True)
        os.makedirs(Path(dst_dir) / "test" / label, exist_ok=True)

    dataset = {
        "train": glob.glob(src_dir + "/train/images/*.jpg", recursive=True),
        "val": glob.glob(src_dir + "/val/images/*.jpg", recursive=True),
        "test": glob.glob(src_dir + "/test/images/*.jpg", recursive=True),
    }

    for datatype, image_file_paths in dataset.items():
        for path in tqdm(image_file_paths):

            image = cv2.imread(path)
            with open(path.replace("images", "labels").replace(".jpg", ".txt")) as f:
                annots = f.readlines()

            for annot in annots:

                elements = annot.split(" ")
                label = class_names[int(elements[0])]

                c_x = float(elements[1])
                c_y = float(elements[2])
                w = float(elements[3])
                h = float(elements[4])

                bbox = [c_x - 0.5 * w, c_y - 0.5 * h, c_x + 0.5 * w, c_y + 0.5 * h]

                bbox = np.clip(bbox, 0.0, 1.0)

                if (
                    bbox[0] >= 1.0
                    or bbox[1] >= 1.0
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                ):
                    continue

                w = image.shape[1]
                h = image.shape[0]

                croppedImg = image[
                    int(bbox[1] * h) : int(bbox[3] * h),
                    int(bbox[0] * w) : int(bbox[2] * w),
                ]

                if croppedImg.shape[0] <= 0 or croppedImg.shape[1] <= 0:
                    continue

                # croppedImg = cv2.resize(make_square_image(croppedImg), (64,64))
                croppedImg, _ = make_square_image(croppedImg)

                filename = f"{uuid.uuid1()}.jpg"
                cv2.imwrite(
                    str(Path(dst_dir) / datatype / label / filename), croppedImg
                )


def remap_classes(
    dataset_directory="",
    src_indices=["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"],
    #             0    1    2    3    4    5    6    7    8    9    10   11
    dst_indices=[
        "B",
        "K",
        "N",
        "P",
        "Q",
        "R",
        "board",
        "mask",
        "b",
        "k",
        "n",
        "p",
        "q",
        "r",
    ],
):

    with open(os.path.join(dataset_directory, "data.yaml"), "r") as f:
        data_str = f.read()

    with open(os.path.join(dataset_directory, "data.yaml"), "w") as f:
        data_str = re.sub(r"names: .*", f"names: {dst_indices}", data_str)
        data_str = re.sub(r"nc: .*", f"nc: {len(dst_indices)}", data_str)
        f.write(data_str)

    labels_files = glob.glob(dataset_directory + "/*/labels/*.txt")
    for label_file in labels_files:
        with open(label_file, "r") as f:
            labels = f.readlines()

        annots_txt = ""
        for i, line in enumerate(labels):
            label_index = line.find(" ")
            label = src_indices[int(line[:label_index])]
            new_label = dst_indices.index(label)
            annots_txt += str(new_label) + line[label_index:]

        with open(label_file, "w") as f:
            f.write(annots_txt)


# merge([
#         "/workspace/ChessLink/data/dataset_test_CL15",
#         "/workspace/ChessLink/data/dataset_test_CL16",
#         "/workspace/ChessLink/data/dataset_test_CL17",
#         "/workspace/ChessLink/data/dataset_test_CL18",],
#     "/workspace/ChessLink/data/dataset_test_CL_merge")

# merge_yolo([
#         "/workspace/ChessLink/data/dataset_yolo_18",
#         "/workspace/ChessLink/data/dataset_yolo_23",
#         "/workspace/ChessLink/data/dataset_yolo_24",
#         "/workspace/ChessLink/data/dataset_yolo_25",
#         "/workspace/ChessLink/data/dataset_yolo_26",],
#     "/workspace/ChessLink/data/dataset_yolo_merge")

# merge_yolo([
#         "/workspace/ChessLink/data/dataset_yolo_merge",
#         "/workspace/ChessLink/data/CL.v4i.yolov8"],
#     "/workspace/ChessLink/data/dataset_yolo_merge_w_real")

# gen_yolo_annots_seg()

# gen_yolo_annots_seg_with_pieces(
#     data_directory="/workspace/ChessLink/data/dataset_test_CL24",
#     dst_dir="/workspace/ChessLink/data/dataset_yolo_25",
#     diff_color=False,
#     diff_pieces=False,
#     use_board=False,
#     overwrite=False
# )

# gen_yolo_annots(
#     data_directory="/workspace/ChessLink/data/dataset_test_CL26",
#     dst_dir="/workspace/ChessLink/data/dataset_yolo_merge",
#     diff_color=True,
#     overwrite=False
# )

# gen_pieces_dataset()

# gen_pieces_dataset()
# extract_kings_queens()
# chessred_2_yolo()

# files = glob.glob("/workspace/ChessLink/data/dataset_test_CL18/*.jpg")
# visualize_annots(random.choice(files))
# visualize_annots("/workspace/ChessLink/data/dataset_test_CL18/data_c2127aa3-8844-11ee-a2c1-a036bc2aad3a.jpg")


# visualize_annots_yolo("/workspace/ChessLink/data/dataset_yolo_23")

# convert_roboflow(
#     "/workspace/ChessLink/data/CL.v9i.yolov8_original",
#     "/workspace/ChessLink/data/CL.v9i.yolov8",
# )
merge_yolo(
    [
        "/workspace/ChessLink/data/dataset_yolo_merge",
        "/workspace/ChessLink/data/CL.v9i.yolov8",
    ],
    "/workspace/ChessLink/data/dataset_yolo_merge_w_real_5k",
)

# remap_classes("/workspace/ChessLink/data/dataset_yolo_merge_remapped")

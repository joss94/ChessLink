import glob
import os
import shutil
import cv2
import numpy as np
import json
from pathlib import Path
import random

from multiprocessing import Pool

from tqdm import tqdm

import uuid

from utils import make_square_image, crop_board

import io
import imageio.v2 as imageio


def clean(dataset_path):

    path=dataset_path + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)

    # Remove images with no GT
    images_with_no_GT = []
    for path in image_file_paths:
        json_path = path.replace(".jpg", ".json")
        if not os.path.exists(json_path):
            images_with_no_GT.append(path)

    print(f"Will remove {len(images_with_no_GT)}/{len(image_file_paths)} images with no GT")
    for path in images_with_no_GT:
        os.remove(path)

def merge(datasets, dataset_dst):
    os.makedirs(dataset_dst, exist_ok=True)

    image_file_paths = []
    for dataset in datasets:
        dataset_images = glob.glob(dataset + '/*.jpg',recursive=True)
        print(len(dataset_images))
        image_file_paths.extend(dataset_images)

    for image_path in tqdm(image_file_paths):
        data_name = Path(image_path).stem
        shutil.copy(image_path, os.path.join(dataset_dst, f"{data_name}.jpg"))
        shutil.copy(image_path.replace(".jpg", ".json"), os.path.join(dataset_dst, f"{data_name}.json"))

def merge_yolo(datasets, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir)/"train", exist_ok=True)
    os.makedirs(Path(dst_dir)/"train"/"images", exist_ok=True)
    os.makedirs(Path(dst_dir)/"train"/"labels", exist_ok=True)

    image_file_paths = []
    for dataset in datasets:
        for image_path in tqdm(glob.glob(dataset + '/train/images/*.jpg')):
            data_name = Path(image_path).stem
            label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")
            dst_img_path = os.path.join(Path(dst_dir)/"train"/"images", f"{data_name}.jpg")
            dst_label_path = os.path.join(Path(dst_dir)/"train"/"labels", f"{data_name}.txt")
            if not Path(dst_img_path.exists()):
                shutil.copy(image_path, dst_img_path)
                shutil.copy(label_path, dst_label_path)

    with open(Path(dst_dir) / "data.yaml", 'w+') as f:
        f.write("train: ../train/images\n")
        f.write("val: ../../chessred_test_yolo/images\n")
        f.write("test: ../../chessred_test_yolo/images\n")
        f.write(f"nc: 12\n")
        f.write(f"names: ['p', 'n', 'b', 'r', 'q', 'k', 'p', 'n', 'b', 'r', 'q', 'k']\n")

        f.write(f"augment: True\n")

def visualize_annots(img_path):

    img = cv2.imread(img_path)
    H = img.shape[0]
    W = img.shape[1]


    with open(img_path.replace(".jpg",".json")) as f:
        annots = json.loads(f.read())

    for piece in annots["pieces"]:
        box = np.array(piece["bbox"])
        label=piece["piece"]
        cv2.rectangle(img,
            (int(box[0]* W), int(box[1] * H)),
            (int(box[2]* W), int(box[3] * H)),
            (255,255,255),
            max(1, int(img.shape[0] / 500))
        )
        cv2.putText(
            img,
            # f'{pieces[i]["score"]:.2f}',
            label,
            org = (int(box[0] * W), int(box[1] * H)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4 * img.shape[0] / 500,
            color=(0,255,0),
            thickness=max(1, int(img.shape[0] / 500))
        )
    cv2.imwrite("/workspace/ChessLink/visu.jpg", img)

def visualize_annots_yolo(dataset_path):

    files = glob.glob(f"{dataset_path}/*/*/*.jpg")
    img_path = Path(np.random.choice(files))

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

        cv2.rectangle(img,
            (int((c_x - 0.5 * w) * W), int((c_y - 0.5 * h) * H)),
            (int((c_x + 0.5 * w) * W), int((c_y + 0.5 * h) * H)),
            (255,255,255),
            max(1, int(img.shape[0] / 500))
        )
        cv2.putText(
            img,
            # f'{pieces[i]["score"]:.2f}',
            label,
            org = (int(c_x * W), int(c_y * H)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4 * img.shape[0] / 500,
            color=(0,255,0),
            thickness=max(1, int(img.shape[0] / 500))
        )
    cv2.imwrite("/workspace/ChessLink/visu.jpg", img)


def gen_yolo_annot(path, dst_dir, train_split, val_split, class_names, diff_color, overwrite):

    annots_path = Path(dst_dir) / "train" / "labels" / (str(Path(path).stem) + ".txt")
    img_path = Path(dst_dir) / "train" / "images" / (str(Path(path).stem) + ".jpg")

    # Image was already parsed, don't do it again
    if not overwrite and (img_path.exists() and annots_path.exists()):
        return 0

    img = cv2.imread(path)
    h, w, _ = img.shape

    with open(path.replace(".jpg",".json")) as f:
        annots = json.loads(f.read())

    board_poly = np.array(annots["board"])
    board_poly[:,0] *= w
    board_poly[:,1] *= -1
    board_poly[:,1] += 1
    board_poly[:,1] *= h
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

        if box[0]>=1.0 or box[2]<=0 or box[1]>=1.00 or box[3]<=0:
            continue

        label = piece["piece"]
        if not diff_color:
            label = label.lower()
        p_class = class_names.index(label)
        center_x = 0.5 * (box[0] + box[2])
        center_y = 0.5 * (box[1] + box[3])
        width = abs(box[2] - box[0])
        height = abs(box[1] - box[3])
        annots_txt += f'{p_class} {center_x} {center_y} {width} {height}\n'

    # shutil.copyfile(path, str(img_path))
    # img_cropped = jpegBlur(img_cropped, np.random.randint(20, 80))
    cv2.imwrite(str(img_path), img_cropped)

    with open(annots_path, 'w+') as f:
        f.write(annots_txt)

    return 0

def gen_yolo_annots(data_directory="/workspace/ChessLink/data/dataset_test_CL21", dst_dir="/workspace/ChessLink/data/dataset_yolo_21", diff_color=True, overwrite=False):

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir)/"train", exist_ok=True)
    os.makedirs(Path(dst_dir)/"train"/"images", exist_ok=True)
    os.makedirs(Path(dst_dir)/"train"/"labels", exist_ok=True)

    path=data_directory + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]
    # image_file_paths = image_file_paths[:100]

    train_split = 0.9
    val_split = 1.0 - train_split

    if diff_color:
        class_names = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    else:
        class_names = ['p', 'n', 'b', 'r', 'q', 'k']

    params = [
        (
            image_file_paths[i],
            dst_dir,
            train_split,
            val_split,
            class_names,
            diff_color,
            overwrite
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

    with open(Path(dst_dir) / "data.yaml", 'w+') as f:
        f.write("train: ../train/images\n")
        f.write("val: ../../chessred_test_yolo/images\n")
        f.write("test: ../../chessred_test_yolo/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']\n")

        f.write(f"augment: True\n")
        # f.write(f"mosaic: 0.0\n")

def extract_kings_queens(data_directory="/workspace/ChessLink/data/dataset_test_CL5", dst_dir="/workspace/ChessLink/data/dataset_qk"):

    os.makedirs(dst_dir, exist_ok=True)

    path=data_directory + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)

    i=0
    for path in tqdm(image_file_paths):

        with open(path.replace(".jpg",".json")) as f:
            annots = json.loads(f.read())

        image = cv2.imread(path)

        H = image.shape[0]
        W = image.shape[1]

        for piece in annots["pieces"]:
            box = piece["bbox"]
            box = np.clip(box, 0.0, 1.0)

            if box[0]>=W or box[2]<=0 or box[1]>=H or box[3]<=0:
                continue

            if piece["piece"].lower() not in ["q", "k"]:
                continue

            # print(box)
            piece_img = image[box[1]:box[3], box[0]:box[2]]

            name = f'queen_{i}.jpg' if piece["piece"].lower() == "q" else f'king_{i}.jpg'

            cv2.imwrite(str(Path(dst_dir) / name), piece_img)
            i+=1

def gen_yolo_annots_seg(data_directory="/workspace/ChessLink/data/dataset_test_CL15", dst_dir="/workspace/ChessLink/data/dataset_yolo_seg_3"):

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir)/"train", exist_ok=True)
    os.makedirs(Path(dst_dir)/"train"/"images", exist_ok=True)
    os.makedirs(Path(dst_dir)/"train"/"labels", exist_ok=True)
    os.makedirs(Path(dst_dir)/"valid", exist_ok=True)
    os.makedirs(Path(dst_dir)/"valid"/"images", exist_ok=True)
    os.makedirs(Path(dst_dir)/"valid"/"labels", exist_ok=True)
    os.makedirs(Path(dst_dir)/"test", exist_ok=True)
    os.makedirs(Path(dst_dir)/"test"/"images", exist_ok=True)
    os.makedirs(Path(dst_dir)/"test"/"labels", exist_ok=True)

    path=data_directory + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]
    # image_file_paths = image_file_paths[:1000]

    train_split = 0.8
    val_split = 0.2

    for path in tqdm(image_file_paths):

        rand = np.random.uniform(0, 1)

        if rand < train_split:
            annots_path = Path(dst_dir) / "train" / "labels" / (str(Path(path).stem) + ".txt")
            img_path = Path(dst_dir) / "train" / "images" / Path(path).name
        elif rand < train_split + val_split:
            annots_path = Path(dst_dir) / "valid" / "labels" / (str(Path(path).stem) + ".txt")
            img_path = Path(dst_dir) / "valid" / "images" / Path(path).name
        else:
            annots_path = Path(dst_dir) / "test" / "labels" / (str(Path(path).stem) + ".txt")
            img_path = Path(dst_dir) / "test" / "images" / Path(path).name

        with open(path.replace(".jpg",".json")) as f:
            annots = json.loads(f.read())

        with open(annots_path, 'w+') as f:

            annot = "0"

            board_corners = [
                annots["board"][0],
                annots["board"][8],
                annots["board"][80],
                annots["board"][72],
            ]

            for corner in board_corners:
                annot += f" {corner[0]} {1.0-corner[1]}"
            f.write(f'{annot}\n')

        cv2.imwrite(str(img_path), cv2.resize(cv2.imread(path), (640, 640)))

        with open(Path(dst_dir) / "data.yaml", 'w+') as f:
            f.write("train: ../train/images\n")
            f.write("val: ../valid/images\n")
            f.write("test: ../test/images\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['board']\n")
            f.write(f"augment: True\n")

def chessred_2_yolo(src_dir="/workspace/ChessLink/data/chessred_test", dst_dir="/workspace/ChessLink/data/chessred_test_yolo"):

    class_indices = [6, 9, 7, 8, 10, 11, 0, 3, 1, 2, 4, 5, 12]

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(Path(dst_dir)/"images", exist_ok=True)
    os.makedirs(Path(dst_dir)/"labels", exist_ok=True)

    path=src_dir + '/*/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)

    with open("./data/chessred_test/annotations.json") as f:
        annots = json.load(f)

    for path in tqdm(image_file_paths):

        annots_path = Path(dst_dir) / "labels" / (str(Path(path).stem) + ".txt")
        img_path = Path(dst_dir) / "images" / (str(Path(path).stem) + ".jpg")

        annot = next(a for a in annots["images"] if a["file_name"] == Path(path).name)
        img_id = annot['id']
        corners_annot = next(c for c in annots["annotations"]["corners"] if c["image_id"] == img_id)
        corners = np.int32([
            corners_annot["corners"]["top_left"],
            corners_annot["corners"]["top_right"],
            corners_annot["corners"]["bottom_left"],
            corners_annot["corners"]["bottom_right"]
            ])

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
            center_x = box[0] + 0.5 * box[2]
            center_y = box[1] + 0.5 * box[3]
            width = box[2]
            height = box[3]
            annots_txt += f'{p_class} {center_x} {center_y} {width} {height}\n'

        # shutil.copyfile(path, str(img_path))
        cv2.imwrite(str(img_path), img_cropped)

        with open(annots_path, 'w+') as f:
            f.write(annots_txt)

def gen_pieces_dataset(src_dir="/workspace/CL/data/dataset5", dst_dir="/workspace/CL/data/dataset_pieces", target_size=(64, 64)):
    os.makedirs(dst_dir, exist_ok=True)

    path=src_dir + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]

    class_names = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

    for path in tqdm(image_file_paths):

        image=cv2.imread(path)
        with open(path.replace(".jpg",".json")) as f:
            annots =  json.loads(f.read())

        for piece in annots["pieces"]:

            bbox = np.array(piece["bbox"])

            w = abs(bbox[2] - bbox[0])
            h = abs(bbox[1] - bbox[3])

            bbox[0] -= 0.1 * w
            bbox[2] += 0.1 * w

            if bbox[0] <= 0 or bbox[2] >= 1.0 or bbox[1] <= 0 or bbox[3] >= 1.0:
                continue

            tmp = bbox[1]
            bbox[1] = 1.0 - bbox[3] - 0.1 * h
            bbox[3] = 1.0 - tmp + 0.1 * h

            bbox = np.clip(bbox, 0.0, 1.0)

            w = image.shape[1]
            h = image.shape[0]

            croppedImg = image[int(bbox[1]*h):int(bbox[3]*h), int(bbox[0]*w):int(bbox[2]*w)]
            croppedImg = cv2.resize(croppedImg, (64,64))

            piece = piece["piece"]
            label = [int(piece == p) for p in class_names]

            filename = f'{class_names.index(piece)}_{uuid.uuid1()}.jpg'
            cv2.imwrite(str(Path(dst_dir) / filename), croppedImg)
            # label = self.class_names.index(piece["piece"])


def jpegBlur(im,q):
    buf = io.BytesIO()
    imageio.imwrite(buf,im,format='jpg',quality=q)
    s = buf.getbuffer()
    im = imageio.imread(s,format='jpg')
    return im

# merge([
#         "/workspace/ChessLink/data/dataset_test_CL15",
#         "/workspace/ChessLink/data/dataset_test_CL16",
#         "/workspace/ChessLink/data/dataset_test_CL17",
#         "/workspace/ChessLink/data/dataset_test_CL18",],
#     "/workspace/ChessLink/data/dataset_test_CL_merge")

# merge_yolo([
#         "/workspace/ChessLink/data/dataset_yolo_14",
#         "/workspace/ChessLink/data/dataset_yolo_16",
#         "/workspace/ChessLink/data/dataset_yolo_17",
#         "/workspace/ChessLink/data/dataset_yolo_18",],
#     "/workspace/ChessLink/data/dataset_yolo_merge2")

# gen_yolo_annots_seg()

gen_yolo_annots(
    data_directory="/workspace/ChessLink/data/dataset_test_CL21",
    dst_dir="/workspace/ChessLink/data/dataset_yolo_22",
    diff_color=False,
    overwrite=True
)

# gen_pieces_dataset()
# extract_kings_queens()
# chessred_2_yolo()

# files = glob.glob("/workspace/ChessLink/data/dataset_test_CL18/*.jpg")
# visualize_annots(random.choice(files))
# visualize_annots("/workspace/ChessLink/data/dataset_test_CL18/data_c2127aa3-8844-11ee-a2c1-a036bc2aad3a.jpg")

# im = cv2.imread(random.choice(files))
# im = jpegBlur(im, 80)
# cv2.imwrite("test.jpg", im)

# visualize_annots_yolo("/workspace/ChessLink/data/dataset_yolo_19")
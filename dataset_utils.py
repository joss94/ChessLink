import glob
import os
import shutil
import cv2
import numpy as np
import json
from pathlib import Path

from tqdm import tqdm

import uuid

data_augment_directory='./dataset_augment'
data_base_directory='./dataset3'
output_directory = './augment_masks'

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
        dataset_images = [img for img in dataset_images if "mask" not in img]
        image_file_paths.extend(dataset_images)

    for i, image_path in enumerate(image_file_paths):
        mask_path = image_path.replace(".jpg", "_mask.jpg")
        json_path = image_path.replace(".jpg", ".json")

        shutil.copy(image_path, os.path.join(dataset_dst, f"data_{i}.jpg"))
        shutil.copy(mask_path, os.path.join(dataset_dst, f"data_{i}_mask.jpg"))
        shutil.copy(json_path, os.path.join(dataset_dst, f"data_{i}.json"))

def gen_masks(data_directory="/workspace/CL/data/dataset5", regenerate=False):

    path=data_directory + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]

    for i, path in enumerate(image_file_paths):
        mask_path = path.replace('.jpg', '_mask.jpg')

        if not regenerate and os.path.exists(mask_path):
            continue

        image = cv2.imread(path)
        if os.path.exists(path.replace(".jpg",".json")):
            with open(path.replace(".jpg",".json")) as f:
                annots =  json.loads(f.read())
            width = image.shape[1]
            height = image.shape[0]

            mask=np.zeros((height, width, 1), np.uint8)
            corners = annots["board"]
            poly = np.array([corners[0], corners[72], corners[80], corners[8]])
            poly[:,0] *= width
            poly[:,1] = (1.0 - poly[:,1]) * height
            poly = poly.astype(np.int32)

            borders = [
                -min(np.min(poly[:,0]), 0),
                max(np.max(poly[:,0]), width) - width,
                -min(np.min(poly[:,1]), 0),
                max(np.max(poly[:,1]), height) - height
                ]

            mask = cv2.copyMakeBorder(mask, borders[2], borders[3], borders[0], borders[1], borderType=cv2.BORDER_CONSTANT, value = (0,0,0))
            poly[:,0] += borders[0]
            poly[:,1] += borders[2]

            cv2.fillPoly(mask, pts = [poly], color=(255, 255,255))
            mask = mask[borders[1]:borders[1] + height, borders[0]:borders[0] + width]

            print(f"Saving mask at {path.replace('.jpg', '_mask.jpg')} ({i}/{len(image_file_paths)})")
            cv2.imwrite(path.replace(".jpg", "_mask.jpg"), mask)



def gen_yolo_annots(data_directory="/workspace/CL/data/dataset5", dst_dir="/workspace/CL/data/dataset_yolo_3"):

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
    # image_file_paths = image_file_paths[:100]

    train_split = 0.9
    val_split = 1.0 - train_split

    class_names = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

    for path in tqdm(image_file_paths):

        rand = np.random.uniform(0, 1)

        with open(path.replace(".jpg",".json")) as f:
            annots = json.loads(f.read())

        image = cv2.imread(path)

        if rand < train_split:
            annots_path = Path(dst_dir) / "train" / "labels" / (str(Path(path).stem) + ".txt")
            img_path = Path(dst_dir) / "train" / "images" / (str(Path(path).stem) + ".jpg")
        elif rand < train_split + val_split:
            annots_path = Path(dst_dir) / "valid" / "labels" / (str(Path(path).stem) + ".txt")
            img_path = Path(dst_dir) / "valid" / "images" / (str(Path(path).stem) + ".jpg")
        else:
            annots_path = Path(dst_dir) / "test" / "labels" / (str(Path(path).stem) + ".txt")
            img_path = Path(dst_dir) / "test" / "images" / (str(Path(path).stem) + ".jpg")

        annots_txt = ""

        board_poly = np.array([
            annots["board"][0],
            annots["board"][8],
            annots["board"][80],
            annots["board"][72],
        ])
        board_poly[:,0] = board_poly[:,0] * image.shape[1]
        board_poly[:,1] = (1.0 - board_poly[:,1]) * image.shape[0]

        [X, Y, W, H] = cv2.boundingRect(np.int32(board_poly))
        X = max(0, X)
        Y = max(0, Y)
        W = min(W, image.shape[1] - X - 1)
        H = min(H, image.shape[0] - Y - 1)
        target_ar = 540/960
        ar = H/W
        if ar < target_ar-0.01:
            dh = target_ar * W - H
            H += dh
            Y -= 0.5 * dh
        elif ar > target_ar+0.01:
            dw = H / target_ar - W
            W += dw
            X -= 0.5 * dw

        X = max(0, X)
        Y = max(0, Y)
        W = min(W, image.shape[1] - X - 1)
        H = min(H, image.shape[0] - Y - 1)

        for piece in annots["pieces"]:
            box = piece["bbox"]
            box = np.array([box[0], 1.0-box[3], box[2], 1.0-box[1]])

            box[0] = (box[0] * image.shape[1] - X) / W
            box[2] = (box[2] * image.shape[1] - X) / W
            box[1] = (box[1] * image.shape[0] - Y) / H
            box[3] = (box[3] * image.shape[0] - Y) / H

            box = np.clip(box, 0.0, 1.0)

            if box[0]>=1.0 or box[2]<=0 or box[1]>=1.00 or box[3]<=0:
                continue

            p_class = class_names.index(piece["piece"])
            center_x = 0.5 * (box[0] + box[2])
            center_y = 0.5 * (box[1] + box[3])
            width = abs(box[2] - box[0])
            height = abs(box[1] - box[3])
            annots_txt += f'{p_class} {center_x} {center_y} {width} {height}\n'

        image_cropped = image[int(Y):int(Y+H),int(X):int(X+W)]

        cv2.imwrite(str(img_path), cv2.resize(image_cropped, (640, 640)))
        with open(annots_path, 'w+') as f:
            f.write(annots_txt)

        with open(Path(dst_dir) / "data.yaml", 'w+') as f:
            f.write("train: ../train/images\n")
            f.write("val: ../valid/images\n")
            f.write("test: ../test/images\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: ['black_pawn', 'black_knight', 'black_bishop', 'black_rook', \
                    'black_queen', 'black_king', 'white_pawn', 'white_knight', 'white_bishop', \
                    'white_rook', 'white_queen', 'white_king']\n")

            f.write(f"augment: True\n")
            # f.write(f"mosaic: 0.0\n")

def gen_yolo_annots_seg(data_directory="/workspace/CL/data/dataset5", dst_dir="/workspace/CL/data/dataset_yolo_seg_2"):

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

        cv2.imwrite(str(img_path), cv2.resize(cv2.imread(path), (480, 480)))

        with open(Path(dst_dir) / "data.yaml", 'w+') as f:
            f.write("train: ../train/images\n")
            f.write("val: ../valid/images\n")
            f.write("test: ../test/images\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['board']\n")
            f.write(f"augment: True\n")


def split(data_directory="/workspace/CL/data/dataset3", split = 0.8):
    path=data_directory + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)
    image_file_paths = [img + "\n" for img in image_file_paths if "mask" not in img]

    split_idx = int(split * len(image_file_paths))

    with open(os.path.join(data_directory, "train.txt"), 'w+') as f:
        f.writelines(image_file_paths[:split_idx])
    with open(os.path.join(data_directory, "val.txt"), 'w+') as f:
        f.writelines(image_file_paths[split_idx:])


def prepare_dataset(src_dir="/workspace/CL/data/dataset3", dst_dir="", target_size=(256, 256)):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    path=src_dir + '/*.jpg'
    image_file_paths=glob.glob(path,recursive=True)
    image_file_paths = [img for img in image_file_paths if "mask" not in img]

    for path in tqdm(image_file_paths):
        image=cv2.imread(path)[:,:,0:3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        dst_path = str(Path(dst_dir) / Path(path).name)
        cv2.imwrite(dst_path, image)

        shutil.copy(path.replace(".jpg", ".json"), dst_path.replace(".jpg", ".json"))


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

# merge(["/workspace/CL/dataset_augment", "/workspace/CL/dataset3"], "/workspace/CL/dataset_merge")
# gen_yolo_annots_seg()
# gen_yolo_annots()
# split()
# prepare_dataset("/workspace/CL/data/dataset5", "/workspace/CL/data/dataset5_preprocessed")
# gen_masks()
gen_pieces_dataset()
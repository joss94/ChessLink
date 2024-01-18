import os
import numpy as np

import torch

from torch.utils.data import Dataset
import torchvision.transforms as tf
from PIL import Image
import glob
import cv2
import json




class ChessDataset(Dataset):
    def __init__(self, data_folder, img_size=480):
        print("Initializing dataset")
        self.imgs = glob.glob(os.path.join(data_folder, "images", "*.jpg")) # Create list of images
        self.img_size = img_size

        self.transformImg=tf.Compose([
            tf.ToPILImage(),
            tf.Resize([self.img_size, self.img_size]),
            tf.ColorJitter(brightness=0.5),
            tf.ToTensor(),
            # tf.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])

        self.classes = [
            '',
            'p',
            'n',
            'b',
            'r',
            'q',
            'k',
            'P',
            'N',
            'B',
            'R',
            'Q',
            'K',
        ]

    def __len__(self):
        # return 20
        return len(self.imgs)

    def __getitem__(self, index):

        img_path = self.imgs[index % len(self.imgs)].rstrip()
        flipped = np.random.randint(0, 2)>0

        # read the image
        image=cv2.imread(img_path)

        with open(img_path.replace(".jpg",".txt").replace("images", "labels")) as f:
            annots =  f.readlines()

        boxes = []
        labels=[]
        for line in annots:
            e = line.split(" ")
            p, c_x, c_y, w, h = (int(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4]))

            box = np.array([c_x - 0.5*w, c_y - 0.5*h, c_x + 0.5*w, c_y + 0.5*h])
            box = np.clip(box, 0.0, 1.0)

            if box[0] > box[2] or box[1] > box[3]:
                print(f"INVALID WIDTH OR HEIGHT! {box}")
                assert(False)

            if flipped:
                tmp = box[0]
                box[0] = 1.0 - box[2]
                box[2] = 1.0 - tmp

            if box[2] > 0.0 and box[3] > 0.0 and box[0] < 1.0 and box[1] < 1.0:
                boxes.append(box)
                labels.append(p)
        boxes = np.array(boxes)
        boxes *= self.img_size

        if flipped:
            image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#.astype(np.float32)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) # bounding box to tensor
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # area of the bounding boxes
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) # no crowd instances
        labels = torch.as_tensor(labels, dtype=torch.int64) # labels to tensor

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return self.transformImg(image), target

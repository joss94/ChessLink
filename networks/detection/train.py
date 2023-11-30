import os

import torch
import torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import glob
from pathlib import Path
import torchvision.transforms as tf
import shutil

from .chess_dataset import ChessDataset


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

class DetectNet:

    def __init__(self, train = False, pretrained_path="", device="cpu"):
        self.img_size = 480
        self.classes = ["BG", 'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # get the model using our helper function
        self.model = self.get_object_detection_model(num_classes=len(self.classes))

        self.model.to(self.device)
        if pretrained_path != "":
            self.model.load_state_dict(torch.load(pretrained_path))
        if not train:
            self.model.eval()

    def get_object_detection_model(self, num_classes=3):

        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def infer(self, images, threshold = 0.5):

        transformImg=tf.Compose([
            tf.ToPILImage(),
            tf.Resize([self.img_size, self.img_size]),
            tf.ToTensor(),
            # tf.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            # tf.Normalize(mean=[0,0,0], std=[2.0, 2.0, 2.0])
            ])

        images_tensors = []
        offsets = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Fix aspect ratio
            ar = image.shape[0]/image.shape[1]
            target_ar = 1.0
            offset = [0, 0, 0, 0]
            if ar < target_ar:
                border = int(0.5 * (target_ar * image.shape[1] - image.shape[0]))
                offset = [border, border, 0, 0]
                image = cv2.copyMakeBorder(image, border, border, 0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))
            elif ar > target_ar:
                border = int(0.5 * (image.shape[0] / target_ar - image.shape[1]))
                offset = [0,0, border, border]
                image = cv2.copyMakeBorder(image, 0,0, border, border, cv2.BORDER_CONSTANT, value = (0,0,0))

            offsets.append(offset)
            images_tensors.append(transformImg(image).to(self.device))

        with torch.no_grad():
            outputs = self.model(images_tensors)

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        res = []
        for output, offset in zip(outputs, offsets):

            boxes = output['boxes'].data.numpy()
            scores = output['scores'].data.numpy()
            labels = output['labels'].cpu().numpy()

            boxes /= self.img_size
            boxes = boxes[scores >= threshold]# filter out boxes according to `detection_threshold`

            (h,w) = image.shape[:2]
            roi_width = ((w - offset[3]) - offset[2]) / w
            roi_height = ((h - offset[1]) - offset[0]) / h
            for box in boxes:
                box[0] = (box[0] - offset[2]/w) / roi_width
                box[1] = (box[1] - offset[0]/h) / roi_height
                box[2] = (box[2] - offset[2]/w) / roi_width
                box[3] = (box[3] - offset[0]/h) / roi_height

            res.append((boxes, scores, labels))

        return res



    def train(self, dataset_path):
        dataset = ChessDataset(dataset_path, self.img_size)

        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        test_split = 0.8
        tsize = int(len(dataset)*test_split)
        subset = torch.utils.data.Subset(dataset, indices[:tsize])
        subset_test = torch.utils.data.Subset(dataset, indices[tsize:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            subset, batch_size=10, shuffle=True, num_workers=16,
            collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            subset_test, batch_size=10, shuffle=False, num_workers=16,
            collate_fn=collate_fn)

        print(f"Training set: {len(subset)} items")
        print(f"Validation set: {len(subset_test)} items")


        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params=params,lr=1e-3)
        # optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.5)
        n_epochs = 1000
        show_freq = 20
        eval_freq = 400
        total_batch_index = 0

        print("Starting training...")
        min_val_loss = 1e6
        for epoch in range(n_epochs): # Training loop

            total_loss = 0.0
            total_items = 0
            for batch_idx, (images, targets) in enumerate(data_loader):

                total_batch_index += 1

                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                images = list(image.to(self.device) for image in images)

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                losses.backward()
                optimizer.step()

                if batch_idx > 0 and batch_idx % show_freq == 0:
                    print (f'    Step {batch_idx}, Loss: {loss_value:.4f}', end='\r')

                total_loss += loss_value
                total_items += 1

                if (batch_idx > 0 and batch_idx % eval_freq == 0) or batch_idx == len(data_loader) - 1:
                    val_loss = self.evaluate(data_loader_test)
                    print(f"(batch {total_batch_index}) Train: {total_loss / total_items:.4f}   -   Val: {val_loss:.4f}")

                    model_path = os.path.join("/workspace/ChessLink/model/detection/", "latest.torch")
                    torch.save(self.model.state_dict(), model_path)

                    if val_loss < min_val_loss:
                        shutil.copy(model_path, model_path.replace("latest", "best"))

            print(f"\nEpoch {epoch}/{n_epochs}: {total_loss / total_items:.4f}")

            # update the learning rate
            lr_scheduler.step()


    def evaluate(self, data_loader):
        total_val_loss = 0.0
        total_val_items = 0
        with torch.no_grad():
            # self.model.eval()
            for _, (images, targets) in enumerate(data_loader):

                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                images = list(image.to(self.device) for image in images)

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                total_val_loss += loss_value
                total_val_items += 1

        return total_val_loss / total_val_items



    def draw_boxes(self, img, scores, boxes, labels):
        # get all the predicited class names

        boxes = boxes.copy()

        h = img.shape[0]
        w = img.shape[1]

        # draw the bounding boxes and write the class name on top of it
        for label, score, box in zip(labels, scores, boxes):
            labels_names = ["BG", "b", "W"]
            labels_names = ["BG", 'p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
            class_name = labels_names[label]
            color = (0, 255, 0) if class_name.isupper() else (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * img.shape[1]/1080
            font_thickness = 1

            cv2.rectangle(img,
                        (int(box[0]*w), int(box[1]*h)),
                        (int(box[2]*w), int(box[3]*h)),
                        color, 1)

            cv2.putText(img, f"{score:.2f}",
                        (int(box[0]*w), int((box[1])*h)),
                        font, font_scale, color,
                        font_thickness, lineType=cv2.LINE_AA)


            pos = (int(box[0]*w), int((box[3])*h))
            text_size, _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(img, (pos[0], pos[1] - text_h), (pos[0] + text_w, pos[1]), (255,255,255), -1)
            cv2.putText(img, class_name, pos, font, font_scale, color, font_thickness, lineType=cv2.LINE_AA)


    def test(self, modelPath, testFolder):
        ListImages = glob.glob(os.path.join(testFolder, "*.jpg")) # Create list of images

        inputs = []
        for imgPath in ListImages:

            print(f"Processing {imgPath}")

            img=cv2.imread(imgPath)[:,:,0:3]

            max_height = 1080
            if img.shape[0] > max_height:
                img = cv2.resize(img, (int((max_height / img.shape[0]) * img.shape[1]), max_height))
            inputs.append(img)

        outputs = self.infer(inputs)

        for img_path, img, output in zip(ListImages, inputs, outputs):
            scores, boxes, labels = output
            self.draw_boxes(img, scores, boxes, labels)
            cv2.imwrite(f"/workspace/CL/output/{Path(img_path).stem}_detect.jpg", img.astype(np.int32))

    def visualize_train(self):
        main_folder = "/workspace/ChessLink/data"
        train_folder = "dataset_test_CL6"
        dataset = ChessDataset(os.path.join(main_folder, train_folder))

        (data_, target_) = dataset[np.random.randint(0, 1e6)]
        # (data_, target_) = dataset[0]

        img = data_.cpu().detach().numpy()
        img = img.swapaxes(0, 2).swapaxes(0, 1)
        img = (img*255).astype(np.int32).copy()

        boxes=target_["boxes"].cpu().detach().numpy()
        boxes /= img.shape[0]
        labels=target_["labels"].cpu().detach().numpy()
        scores=[1.0 for label in labels]
        self.draw_boxes(img, scores, boxes, labels)

        cv2.imwrite("/workspace/ChessLink/train.jpg", img)

if __name__ == "__main__":
    Net = DetectNet(train=True, device="cuda:1")#, pretrained_path="/workspace/ChessLink/model/detection/latest.torch")
    # Net.visualize_train()
    # Net.test("/workspace/CL/model/detection/5.torch", "/workspace/CL/data/test_images")
    Net.train("/workspace/ChessLink/data/dataset_yolo_18")
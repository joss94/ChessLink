import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import glob
from pathlib import Path
import json
import random
from PIL import Image


#----------------------------------------------Transform image-------------------------------------------------------------------
IMG_SIZE=64
transform=tf.Compose([tf.ToPILImage(),tf.Resize((IMG_SIZE, IMG_SIZE))])

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


class PieceDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        self.listImages = glob.glob(os.path.join(data_folder, "*.jpg")) # Create list of images
        # self.listImages = [img for img in self.listImages if "mask" not in img]
        self.class_names = ['q', 'k']

    def __len__(self):
        return len(self.listImages)

    def __getitem__(self, index):

        path = self.listImages[index]
        image = cv2.imread(path)

        H = image.shape[0]
        W = image.shape[1]

        image = cv2.resize(image[0:W,0:W], (IMG_SIZE, IMG_SIZE))
        image = np.swapaxes(image, 0, 2)

        label = [0, 1] if 'queen' in str(Path(path).stem) else [1, 0]

        return image, label




class ChessNet(nn.Module):
    def __init__(self, model_path="", device_id=-1, train=False):
        super(ChessNet, self).__init__()

        self.img_size=(12)

        self.class_names = ["q", "k"]
        self.device = torch.device(f'cuda:{device_id}') if (torch.cuda.is_available()and device_id>=0) else torch.device('cpu')
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(32768,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

        if train:
            print(self)

        self.to(self.device)
        if model_path != "":
            self.load_state_dict(torch.load(model_path)) # Load trained model

        if not train:
            self.eval()

    def forward(self, x):
        return self.network(x)

    # ------------- Infer on test image ----------------------------------------------------
    def infer(self, images):

        for i, image in enumerate(images):
            W = image.shape[1]
            images[i] = cv2.resize(image[0:W,0:W], (IMG_SIZE, IMG_SIZE))

        data_ = torch.Tensor(np.array(images)).to(self.device).swapaxes(1,3)#.swapaxes(2,3)
        outputs = self(data_)
        return [outputs.cpu().detach().numpy()]


    def train_network(self):
        main_folder = "/workspace/ChessLink"
        train_folder = "data/dataset_qk"

        batch_size=64
        Learning_Rate=1e-3
        n_epochs = 1000
        show_freq = 1
        test_freq = 150

        dataset = PieceDataset(os.path.join(main_folder, train_folder))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        #--------------Load and set net and optimizer-------------------------------------
        # Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
        # Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
        optimizer=torch.optim.Adam(params=self.parameters(),lr=Learning_Rate) # Create adam optimizer

        criterion = nn.CrossEntropyLoss()

        #----------------Train--------------------------------------------------------------------------

        n_batches = len(train_loader)
        current_loss = 0.0
        for itr in range(n_epochs): # Training loop
            total_loss = 0.0
            total_items = 1
            for batch_idx, (data_, target_) in enumerate(train_loader):

                # data_ = list(d.to(self.device) for d in data_)
                # target_ = [t.to(self.device) for t in target_]

                data_ = torch.Tensor([d for d in data_]).to(self.device)
                target_ = torch.Tensor([p for p in target_]).to(self.device)

                # print(data_.size())

                #data_, target_ = data_.to(device), target_.to(device)# on GPU
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()

                current_loss = 0.9 * current_loss + 0.1 * loss.item()
                # print statistics
                if (batch_idx+1) % show_freq == 0:
                    print (f'    Step {batch_idx}/{n_batches} - Loss: {current_loss:.4f}', end = '\r')

                total_loss += loss.item()
                total_items += 1

                if (total_items+1) % test_freq == 0:

                    print(f"\nEpoch {itr}/{n_epochs}: {total_loss / total_items:.4f}")

                    model_path = os.path.join(main_folder, "model/classif/latest.torch")
                    torch.save(self.state_dict(), model_path)
                    #est(model_path, testFolder=os.path.join(main_folder, "test_images"))




def visualize_train():
    main_folder = "/workspace/CL"
    train_folder = "dataset5"
    dataset = PieceDataset(os.path.join(main_folder, train_folder))

    (data_, target_) = dataset[0]

    img = data_.cpu().detach().numpy()
    img = img.swapaxes(0, 2).swapaxes(0, 1)
    img = (img*255).astype(np.int32).copy()
    print(img.shape)


    gt = target_.cpu().detach().numpy()
    idx = np.argmax(gt)
    label = ["'black", 'white', 'empty'][idx]

    cv2.putText(
        img,
        label,
        org = (int(0.1 * img.shape[1]), int(0.9 * img.shape[0])),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.5,
        color=(0,255,0)
        )
    cv2.imwrite("/workspace/CL/train.jpg", img)



if __name__ == "__main__":
    # test("/workspace/CL/model/classif/999.torch", "/workspace/CL/test_images")
    net = ChessNet("", 0, True)
    net.train_network()
    # visualize_train()
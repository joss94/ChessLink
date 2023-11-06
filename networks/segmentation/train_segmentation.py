import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import glob
from pathlib import Path


class SegmentNet():

    def __init__(self, train=False, pretrained_path="", device="cpu"):
        self.img_size = 512
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((self.img_size,self.img_size)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((self.img_size,self.img_size),tf.InterpolationMode.NEAREST),tf.ToTensor()])

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
        self.model = self.model.to(self.device)  # Set net to GPU or CPU
        if pretrained_path != "":
            self.model.load_state_dict(torch.load(pretrained_path))
        if not train:
            self.model.eval() # Set to evaluation mode

    def ReadRandomImage(self): # First lets load random image and  the corresponding annotation
        idx=np.random.randint(0,len(self.img_list)) # Select random image
        Img=cv2.imread(self.img_list[idx])[:,:,0:3]
        Mask =  cv2.imread(self.img_list[idx].replace(".jpg","_mask.jpg"),0)
        Img=self.transformImg(Img)
        Mask=self.transformAnn(Mask)
        return Img,Mask

    def LoadBatch(self, batch_size): # Load batch of images
        images = torch.zeros([batch_size,3,self.img_size,self.img_size])
        masks = torch.zeros([batch_size, self.img_size, self.img_size])
        for i in range(batch_size):
            images[i],masks[i]=self.ReadRandomImage()
        return images, masks

    def infer(self, images):

        masks = []
        for image in images:
            height_orgin , widh_orgin ,d = image.shape # Get image original size
            Img = self.transformImg(image)  # Transform to pytorch
            Img = torch.autograd.Variable(Img, requires_grad=False).to(self.device).unsqueeze(0)
            with torch.no_grad():
                Prd = self.model(Img)['out']  # Run net
            Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0]) # Resize to origninal size

            seg = torch.argmax(Prd, 0).cpu().detach().numpy()  # Get  prediction classes
            seg = np.expand_dims(seg, axis=2)

            masks.append(seg)

        return masks


    def train(self):
        main_folder = "/workspace/CL"
        train_folder = "data/dataset5"

        Learning_Rate=1e-5
        n_steps = 10000
        test_freq = 500
        batch_size = 4

        self.img_list = glob.glob(os.path.join(main_folder, train_folder, "*.jpg")) # Create list of images
        self.img_list = [img for img in self.img_list if "mask" not in img]

        #--------------Load and set net and optimizer-------------------------------------
        optimizer=torch.optim.Adam(params=self.model.parameters(),lr=Learning_Rate) # Create adam optimizer

        #----------------Train--------------------------------------------------------------------------
        for itr in range(n_steps): # Training loop
            images,masks=self.LoadBatch(batch_size) # Load taining batch
            images=torch.autograd.Variable(images,requires_grad=False).to(self.device) # Load image
            masks = torch.autograd.Variable(masks, requires_grad=False).to(self.device) # Load annotation
            Pred=self.model(images)['out'] # make prediction
            self.model.zero_grad()
            criterion = torch.nn.CrossEntropyLoss() # Set loss function
            Loss=criterion(Pred,masks.long()) # Calculate cross entropy loss
            Loss.backward() # Backpropogate loss
            optimizer.step() # Apply gradient descent change to weight
            print(f"({itr}) Loss= {Loss.data.cpu().numpy()}")
            if itr % test_freq == 0: #Save model weight once every 60k steps permenant file
                model_path = os.path.join(main_folder, "model/", str(itr) + ".torch")
                print("Saving Model at " + model_path)
                torch.save(self.model.state_dict(), model_path)
                # self.test(testFolder=os.path.join(main_folder, "data/test_images"))

    def test(self, testFolder):
        img_list = glob.glob(os.path.join(testFolder, "*.jpg")) # Create list of images
        img_list = [img for img in img_list if "mask" not in img]

        for imgPath in img_list:
            print(imgPath)
            print(cv2.imread(imgPath).shape)


        masks = self.infer([cv2.imread(imgPath) for imgPath in img_list])

        for imgPath, mask in zip(img_list, masks):
            original = cv2.imread(imgPath)
            masked_img = original.astype(np.float32) * (mask.astype(np.float32) + 0.2)
            save_path = imgPath.replace(".jpg", f"_mask.jpg")
            cv2.imwrite(save_path, masked_img.astype(np.int32))

if __name__ == "__main__":
    Net = SegmentNet(train=True)
    # test("/workspace/CL/model/4000.torch", "/workspace/CL/test_images")
    Net.train()
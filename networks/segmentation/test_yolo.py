from ultralytics import YOLO
import numpy as np
import cv2


# Load a model
model = YOLO('/workspace/CL/runs/segment/train2/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
image_path = '/workspace/CL/data/test_images/1000011472.jpg'
image_path = '/workspace/CL/data/test_images/hand_in_front.png'
image = cv2.imread(image_path)

ar = image.shape[0]/image.shape[1]
target_ar = 540/960
offset = [0, 0, 0, 0]

if ar < target_ar:
    border = int(0.5 * (target_ar * image.shape[1] - image.shape[0]))
    offset = [border, border, 0, 0]
    image = cv2.copyMakeBorder(image, border, border, 0, 0, cv2.BORDER_CONSTANT, value = (0,0,0))
elif ar > target_ar:
    border = int(0.5 * (image.shape[0] / target_ar - image.shape[1]))
    offset = [0,0, border, border]
    image = cv2.copyMakeBorder(image, 0,0, border, border, cv2.BORDER_CONSTANT, value = (0,0,0))

results = model(cv2.resize(image, (480,480)), device=[0])[0]
mask = np.uint8(results.masks.data.cpu().numpy() * 255)
mask = np.swapaxes(mask, 0, 1)
mask = np.swapaxes(mask, 1, 2)

mask = cv2.resize(mask, (image.shape[1],image.shape[0]))
# image = cv2.resize(image, (mask.shape[1],mask.shape[0]))
masked_image = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB), 0.5, 0)
cv2.imwrite("/workspace/CL/yolo_test.jpg", masked_image)
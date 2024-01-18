import numpy as np
import cv2

from networks.classification.train_classif import ChessNet


image_path = '/workspace/ChessLink/cropped.jpg'
img = cv2.imread(image_path)
classif_net = ChessNet("model/classif/latest.torch", device_id=0)


output = classif_net.infer([img])
new_label = "q" if output[0][0][1] > output[0][0][0] else "k"
print(new_label)
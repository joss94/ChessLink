from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os

# load the class label names from disk, one label per line
CLASS_NAMES = ["chessboard"]

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "chessboard_detection"
	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = 1

# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)
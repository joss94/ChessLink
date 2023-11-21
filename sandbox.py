import cv2
import glob
import json
import utils

files = glob.glob("/workspace/ChessLink/data/dataset_test_CL10/*.jpg")
image_path = files[1]
image = cv2.imread(image_path)

with open(image_path.replace(".jpg",".json")) as f:
     annots = json.loads(f.read())

aligned = utils.align_image(image, annots["board"])

cv2.imwrite("./aligned.jpg", aligned)

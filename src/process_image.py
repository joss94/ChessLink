import cv2
import numpy as np

from app.chessboard_parser import ChessboardParser
from utils.image_utils import draw_results_on_image

DEVICE = "cpu"

image_path = "/workspace/jma_test/data/gamestate_test/0020.png"
image_path = "/workspace/jma_test/data/chessred_test/0/G000_IMG040.jpg"
image_path = "/workspace/jma_test/data/test_images/1000011469.jpg"

image = cv2.imread(image_path)

# image = image[:int(image.shape[0]/2),int(image.shape[1]/2):]

VIDEO_PATH = "/workspace/jma_test/data/test_images/caruana_1080p.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 3400)
ret, image = cap.read()

chessboard_parser = ChessboardParser(device=DEVICE, detect_weights="/workspace/jma_test/src/runs/detect/real_mix_scale_newset_bs6415/weights/last.onnx")

results = chessboard_parser.process_images([image])[0]
print(results)
draw_results_on_image(image, results)
cv2.imwrite("./yolo_test.jpg", image)

import cv2
import numpy as np

from app.chessboard_parser import ChessboardParser
from utils.image_utils import draw_results_on_image

DEVICE = 0

image_path = "/workspace/ChessLink/data/gamestate_test/0020.png"
image_path = "/workspace/ChessLink/data/chessred_test/0/G000_IMG040.jpg"
# image_path = '/workspace/ChessLink/data/dataset_test_CL6/data_aaab3a0a-8318-11ee-b9f1-a036bc2aad3a.jpg'
image_path = "/workspace/ChessLink/data/test_images/1000011469.jpg"
# image_path = '/workspace/ChessLink/data/dataset_test_CL3/data_75cfde3a-820f-11ee-8c9c-a036bc2aad3a.jpg'
# image_path = '/workspace/ChessLink/data/dataset_test_CL3/data_8897d096-820f-11ee-b357-a036bc2aad3a.jpg'

image = cv2.imread(image_path)

VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana_1080p.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 7400)
ret, image = cap.read()

chessboard_parser = ChessboardParser(DEVICE)

results = chessboard_parser.process_images([image])[0]
draw_results_on_image(image, results)
cv2.imwrite("/workspace/ChessLink/yolo_test.jpg", image)

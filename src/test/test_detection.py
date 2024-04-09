import cv2
import numpy as np

from utils.image_utils import make_square_image

from networks.detection.YOLO_det_onnx import YOLOv8ONNX

from ultralytics import YOLO

from pathlib import Path



DEVICE = 1
CONF = 0.3
LABELS = ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]

image_path = "/workspace/jma_test/data/gamestate_test/0020.png"
image_path = "/workspace/jma_test/data/chessred_test_yolo/val/images/G006_IMG004.jpg"
# image_path = '/workspace/jma_test/data/dataset_test_CL6/data_aaab3a0a-8318-11ee-b9f1-a036bc2aad3a.jpg'
# image_path = "/workspace/jma_test/data/test_images/1000011469.jpg"
# image_path = '/workspace/jma_test/data/dataset_test_CL3/data_75cfde3a-820f-11ee-8c9c-a036bc2aad3a.jpg'
# image_path = '/workspace/jma_test/data/dataset_test_CL3/data_8897d096-820f-11ee-b357-a036bc2aad3a.jpg'

image = cv2.imread(image_path)

# image = image[int(image.shape[0]/2):,int(image.shape[1]/2):]
# image = image[int(0.2 * image.shape[0]):int(0.8 * image.shape[0]),:int(image.shape[1]/2)]
border = int(image.shape[0] * 0.2)
# image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana_1080p.mp4"
# cap = cv2.VideoCapture(VIDEO_PATH)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 7400)
# ret, image = cap.read()

detect_weights = str(Path("./rmodel/detection/yolo/weights.pt"))
detect_weights = str(Path("./runs/detect/real_mix_scale5/weights/last.pt"))

if detect_weights.endswith("onnx"):
    detect_net = YOLOv8ONNX(detect_weights)
else:
    detect_net = YOLO(detect_weights)

image, _ = make_square_image(image)

image = cv2.resize(image, (1080, 1080))

# ONNX version:
# boxes, labels, scores = self.detect_net(img_cropped)

# Ultralytics version:
detections = detect_net(image, device = DEVICE, verbose=False, conf=CONF)[0]
boxes = np.array([b.cpu().numpy() for b in detections.boxes.xyxyn]).reshape((-1,4))
labels = [int(c.cpu()) for c in detections.boxes.cls]
scores = [s.cpu() for s in detections.boxes.conf]

h, w, _ = image.shape

for box, label in zip(boxes, labels):
    cv2.rectangle(
        image,
        (int(box[0] * w), int(box[1] * h)),
        (int(box[2] * w), int(box[3] * h)),
        (255, 255, 255),
        max(1, int(h / 500)),
    )
    cv2.putText(
        image,
        LABELS[label],
        # piece.symbol(),
        org=(int(box[0] * w), int(box[1] * h)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5 * h / 1000,
        color=(0, 0, 255),
        thickness=max(1, int(h / 500)),
    )

cv2.imwrite("./yolo_test.jpg", image)

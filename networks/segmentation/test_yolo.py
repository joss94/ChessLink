from ultralytics import YOLO
import numpy as np
import cv2
import glob

from utils import make_square_image, crop_board


# Load a model
model = YOLO('/workspace/ChessLink/runs/segment/train25/weights/last.pt')  # load a pretrained model (recommended for training)

image_path = '/workspace/ChessLink/data/test_images/1000011472.jpg'
image_path = '/workspace/ChessLink/data/test_images/sequence/PXL_20230823_135359218.jpg'
# image_path = np.random.choice(glob.glob('/workspace/ChessLink/data/dataset_yolo_24/train/images/*.jpg'))
# image_path = np.random.choice(glob.glob('/workspace/ChessLink/data/dataset_yolo_24/valid/images/*.jpg'))
image_path = np.random.choice(glob.glob('/workspace/ChessLink/data/chessred_test/*/*.jpg'))
# image_path = '/workspace/CL/data/test_images/hand_in_front.png'
image = cv2.imread(image_path)



VIDEO_PATH = "/workspace/ChessLink/data/test_images/caruana_1080p.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
ret, image = cap.read()


img_cropped = make_square_image(image)[0]
# img_cropped = cv2.resize(image, (640, 640))


results = model(img_cropped, device=[0], conf=.01)[0]
labels = [int(c.cpu()) + 1 for c in results.boxes.cls]
seg_boxes = np.array([b.cpu().numpy() for b in results.boxes.xyxyn]).reshape((-1,4))
results = results.masks.data.cpu().numpy()

mask = np.zeros((640, 640, 3), dtype=np.uint8)

for i in range(results.shape[0]):

    label = labels[i]
    # if label != 1:
    #     continue

    np.random.seed(label)
    np.random.seed(i*5)
    h = np.random.randint(0, 255)
    # h = int((label / 15) * 255)
    s = 255
    v = 255

    # if label == 1:
    #     s = 0
    #     v = 100

    rgb = cv2.cvtColor(np.array([[[h, s,  v]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0]

    piece_mask = results[i, :, :]>0
    mask[results[i, :, :]>0] = rgb

    box = seg_boxes[i]
    cv2.rectangle(mask,
            (int(box[0]*mask.shape[1]), int(box[1]*mask.shape[0])),
            (int(box[2]*mask.shape[1]), int(box[3]*mask.shape[0])),
            (255,255,255),
            max(1, int(h / 500))
        )

# mask = np.swapaxes(mask, 0, 1)
# mask = np.swapaxes(mask, 1, 2)
# print(mask.shape)

mask = cv2.resize(mask, (img_cropped.shape[1],img_cropped.shape[0]))
# image = cv2.resize(image, (mask.shape[1],mask.shape[0]))
gray = cv2.cvtColor(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

alpha = 0.8
masked_image = cv2.addWeighted(gray, alpha, mask, (1.0 - alpha), 0)
cv2.imwrite("/workspace/ChessLink/yolo_test.jpg", masked_image)
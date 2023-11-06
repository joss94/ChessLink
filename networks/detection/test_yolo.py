from ultralytics import YOLO
import numpy as np
import cv2

class_names = [
            'p',
            'n',
            'b',
            'r',
            'q',
            'k',
            'P',
            'N',
            'B',
            'R',
            'Q',
            'K',
        ]

# Load a model
model = YOLO('/workspace/CL/detection/runs/detect/train4/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
image_path = '/workspace/CL/data/test_images/sequence2/Screenshot from 2023-08-25 02-00-59.png'
results = model(image_path, device=[0])[0]
boxes = np.array([b.xyxy.cpu().numpy()[0] for b in results.boxes]).reshape((-1,4))
labels = [class_names[int(b.cls.cpu().numpy()[0])] for b in results.boxes]
confs = [b.conf.cpu().numpy()[0] for b in results.boxes]
print(confs)

image = cv2.imread(image_path)
for conf, label, box in zip(confs, labels, boxes):
    # if conf < 0.5:
    #     continue
    cv2.rectangle(image,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (255,255,255),
        max(1, int(image.shape[0] / 500))
    )
    cv2.putText(
        image,
        # f'{pieces[i]["score"]:.2f}',
        label,
        org = (int(box[0]), int(box[1])),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4 * image.shape[0] / 500,
        color=(0,255,0),
        thickness=max(1, int(image.shape[0] / 500))
    )
cv2.imwrite("/workspace/CL/yolo_test.jpg", image)
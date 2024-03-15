from ultralytics import YOLO

TRAIN = True
RESUME = False

NAME = "test"
DEVICE = 0
DATASET = "/workspace/ChessLink/data/dataset_pieces/"

# Load a model
if TRAIN and not RESUME:
    model = YOLO("yolov8m-cls.pt")  # load a pretrained model (recommended for training)
else:
    model = YOLO(
        f"/workspace/ChessLink/runs/detect/{NAME}/weights/last.pt"
    )

# Train the model with 2 GPUs
if TRAIN:
    results = model.train(
        name=NAME,
        resume=RESUME,
        data=DATASET,
        device=DEVICE,

        batch=8000,
        patience=1000,
        epochs=1000,
        imgsz=128,
        # fraction=0.1
)

metrics = model.val()  # no arguments needed, dataset and settings remembered

metrics.top1
metrics.top5

print(metrics)

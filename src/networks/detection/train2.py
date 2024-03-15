from ultralytics import YOLO

RESUME = False
NAME = "real_mix"
DEVICE = 0
TRAIN = True
DATASET = "/workspace/ChessLink/data/dataset_yolo_merge_w_real_5k/data.yaml"
# DATASET = '/workspace/ChessLink/data/CL.v6i.yolov8/data.yaml'

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/ChessLink/runs/detect/real_mix7/weights/last.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/CL/runs/train/last.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# results = model.train(data='/workspace/ChessLink/data/dataset_yolo_9/data.yaml', epochs=1000, optimizer="Adam", lr0=1e-4, lrf=1e-2, imgsz=640, device=[3])
if TRAIN:
    results = model.train(
        name=NAME,
        resume=RESUME,
        data=DATASET,
        # dropout=0.1,
        epochs=500,
        imgsz=640,
        device=DEVICE,
        # lr0=1e-3,
        # optimizer="SGD",
        mosaic=0.0,
        scale=0.0,
        # hsv_h=0.0,
        # hsv_v=0.0,
        # hsv_s=0.0,
        gaussian_noise=0.3,
        jpg_quality=0.7,
        translate=0.0,
        degrees=5,
        # cls = 0.1,
        # box=20,
        dfl=10,
        # warmup_epochs=30,
        batch=16,
        augment=True,
        patience=1000,
        # fraction=0.05,
        # mixup = 0.3
    )

metrics = model.val(
    device=DEVICE, data=DATASET
)  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

print(metrics)

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/CL/runs/train/last.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# results = model.train(data='/workspace/ChessLink/data/dataset_yolo_9/data.yaml', epochs=1000, optimizer="Adam", lr0=1e-4, lrf=1e-2, imgsz=640, device=[3])
results = model.train(
    data='/workspace/ChessLink/data/dataset_yolo_21/data.yaml',
    # dropout=0.5,
    epochs=1000,
    imgsz=640,
    device=3,
    lr0=1e-2,
    optimizer="SGD",
    mosaic=0.0,
    scale=0.0,
    hsv_h=0.0,
    hsv_v=0.0,
    hsv_s=0.0,
    translate=0.0,
    degrees=0.0,
    # cls = 0.1,
    warmup_epochs=0.5,
    batch=16,
    augment=False,
    # dfl=0.0,
    patience=500
    )
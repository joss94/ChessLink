from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/CL/runs/train/last.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data='/workspace/ChessLink/data/dataset_yolo_seg_3/data.yaml',
    epochs=1000,
    imgsz=640,
    device=1,
    # mosaic=0.0,
    # scale=0.0,
    # hsv_h=0.0,
    # hsv_v=0.0,
    # hsv_s=0.0,
    # translate=0.0,
    # degrees=5.0,
    # cls = 0.1,
    warmup_epochs=0.05,
    batch=16,
    augment=True,
    # dfl=0.0,
    patience=1000
)
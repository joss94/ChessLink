from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/CL/runs/train/last.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# results = model.train(data='/workspace/ChessLink/data/dataset_yolo_9/data.yaml', epochs=1000, optimizer="Adam", lr0=1e-4, lrf=1e-2, imgsz=640, device=[3])
results = model.train(
    data='/workspace/ChessLink/data/dataset_yolo_18/data.yaml',
    # dropout=0.5,
    epochs=200, imgsz=640, device=2, lr0=1e-2, optimizer="SGD", cos_lr=True)
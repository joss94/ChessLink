from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/CL/runs/train/last.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data='/workspace/ChessLink/data/dataset_pieces/',
    epochs=1000,
    imgsz=128,
    device=
    0,
    patience=1000,
    augment=True,
    batch=8000,
    # fraction=0.1
)
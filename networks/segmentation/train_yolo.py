from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO('/workspace/CL/runs/train/last.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/workspace/CL/data/dataset_yolo_seg_2/data.yaml', epochs=100, imgsz=480, device=[2])
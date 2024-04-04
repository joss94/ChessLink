from ultralytics import YOLO

TRAIN = True
RESUME = False

NAME = "real_mix_scale_newset_bs643"
DEVICE = [
    0,
    # 1,
    # 2,
    # 3,
]
# DEVICE='cpu'

DATASET = "/workspace/jma_test/data/dataset_yolo_merge_29_6/data.yaml"

# Load a model
if TRAIN and not RESUME:
    model = YOLO("yolov9c.yaml")
else:
    model = YOLO(
        f"/workspace/jma_test/src/runs/detect/{NAME}/weights/last.pt"
    )
if TRAIN:
    results = model.train(
        name=NAME,
        resume=RESUME,
        data=DATASET,
        device=DEVICE,
        pretrained=True,
        # Training params
        # ---------------------
        batch=64,
        patience=1000,
        epochs=2000,
        imgsz=640,
        # dropout=0.1,
        # lr0=1e-3,
        # optimizer="SGD",
        # cls = 0.1,
        # box=20,
        # dfl=10,
        # warmup_epochs=30,
        # fraction=0.05,
        # Image transforms
        # ---------------------
        translate=0.0,
        scale=0.7,
        degrees=15,
        perspective=0.0005,
        shear=15,
        mosaic=0.0,
        fliplr=0.5,
        flipud=0.0,
        # Pixel transforms
        # ---------------------
        # hsv_h=0.0,
        # hsv_v=0.0,
        # hsv_s=0.0,
        # gaussian_noise= 0.3,
        # jpg_quality= 0.7,
        # Objects transforms
        # ---------------------
        copy_paste=0.5,
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

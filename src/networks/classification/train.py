from ultralytics import YOLO

TRAIN = False
RESUME = False

NAME = "test6"
DEVICE = [
    0,
    # 1,
    # 2,
    # 3,
]
DATASET = "/workspace/jma_test/data/dataset_pieces_3/"

# Load a model
if TRAIN and not RESUME:
    model = YOLO("yolov8m-cls.pt")  # load a pretrained model (recommended for training)
else:
    model = YOLO(
        f"/workspace/jma_test/src/runs/classify/{NAME}/weights/last.pt"
    )

if TRAIN:
    results = model.train(
        name=NAME,
        resume=RESUME,
        data=DATASET,
        device=DEVICE,

        batch=256,
        patience=1000,
        epochs=1000,
        imgsz=256,
        # fraction=0.1,

        translate=0.0,
        scale=0.0,
        degrees=0,
        perspective=0.0000,
        shear=0,
        mosaic=0.0,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.0,

        hsv_h=0.0,
        hsv_v=0.0,
        hsv_s=0.0,
        auto_augment=''
)

metrics = model.val(batch=256)  # no arguments needed, dataset and settings remembered

metrics.top1
metrics.top5

print(metrics)

from ultralytics import YOLO

NAME = "real_mix_scale_hugeset_merged_20k"
TRAIN = True
RESUME = False
PRETRAINED = False

# WEIGHTS = "yolov9c.yaml"
WEIGHTS = "yolov8m.yaml"
# WEIGHTS = f"/workspace/jma_test/src/runs/detect/real_mix_scale_newset_bs643/weights/best.pt"
# WEIGHTS = f"/workspace/jma_test/src/runs/detect/real_mix_scale_hugeset_bs645/weights/last.pt"
# WEIGHTS = f"/workspace/jma_test/src/runs/detect/real_mix_scale_bs64/weights/best.pt"
# WEIGHTS = f"/workspace/jma_test/src/runs/detect/{NAME}/weights/best.pt"

DEVICE = [
    # 0,
    # 1,
    2,
    3,
]
# DEVICE='cpu'

# DATASET = "/workspace/jma_test/data/dataset_yolo_29_3/data.yaml"
DATASET = "/workspace/jma_test/data/dataset_yolo_merge_29_7/data.yaml"
# DATASET = "/workspace/jma_test/data/chessred_test_yolo/data.yaml"
# DATASET = "/workspace/jma_test/data/CL.v11i.yolov8/data.yaml"

# Load a model
model = YOLO(WEIGHTS)
if TRAIN:
    results = model.train(
        name=NAME,
        resume=RESUME,
        data=DATASET,
        device=DEVICE,
        pretrained=PRETRAINED,
        # single_cls=True,
        # Training params
        # ---------------------
        batch=128,
        patience=1000,
        epochs=2000,
        imgsz=640,
        # dropout=0.1,
        # lr0=1e-3,
        optimizer="SGD",
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
        perspective=0.0003,
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
        # copy_paste=0.5,
        # mixup = 0.3

        plots= True,
        workers=16,
        exist_ok = True,
    )

metrics = model.val(
    device=DEVICE,
    data=DATASET,
    plots=True,
    max_det=100

)  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

# print(metrics)

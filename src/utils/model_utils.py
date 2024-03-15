from ultralytics import YOLO


def convert_yolo_to_onnx(path):
    model = YOLO(path)
    model.export(format="onnx")


convert_yolo_to_onnx("/src/model/segment/yolo/weights.pt")
#'/src/model/segment/yolo/weights.onnx'

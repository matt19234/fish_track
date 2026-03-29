import argparse
from ultralytics import YOLO
import os

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("model_path")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

model = YOLO("yolov8n-obb.pt")
model.train(
    data=args.data_path,
    epochs=100,
    imgsz=640,
    device="mps"
)

model.save(args.model_path)
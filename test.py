import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("input_video")
parser.add_argument("output_video")
args = parser.parse_args()

model = YOLO(args.model_path)

cap = cv2.VideoCapture(args.input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

for i in tqdm(range(total_frames), desc="Processing", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, verbose=False)

    if results[0].obb is not None:
        obb = results[0].obb
        confs = obb.conf.cpu().numpy()
        top_k = min(5, len(confs))
        top_indices = np.argsort(confs)[::-1][:top_k]

        for idx in top_indices:
            corners = obb.xyxyxyxy[idx].cpu().numpy().reshape(4, 2).astype(int)
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    # if results[0].obb is not None:
    #     for box in results[0].obb.xyxyxyxy:
    #         corners = box.cpu().numpy().reshape(4, 2).astype(int)
    #         cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    out.write(frame)

cap.release()
out.release()
print("Done")
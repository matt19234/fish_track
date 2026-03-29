import cv2
import os
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("dataset")
args = parser.parse_args()

VIDEO_FILE = args.video
DATASET_DIR = args.dataset
N_FRAMES = 500
TRAIN_COUNT = 400

os.makedirs(DATASET_DIR + "/images/train", exist_ok=True)
os.makedirs(DATASET_DIR + "/images/val", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_FILE)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

indices = [int(i * total_frames / N_FRAMES) for i in range(N_FRAMES)]

frames = []
for idx in tqdm(indices, desc="Extracting frames"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames.append((idx, frame))

cap.release()

random.shuffle(frames)
train_frames = frames[:TRAIN_COUNT]
val_frames = frames[TRAIN_COUNT:]

for i, frame in tqdm(train_frames, desc="Writing train"):
    cv2.imwrite(DATASET_DIR + f"/images/train/frame_{i:06d}.jpg", frame)

for i, frame in tqdm(val_frames, desc="Writing val"):
    cv2.imwrite(DATASET_DIR + f"/images/val/frame_{i:06d}.jpg", frame)

print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")
#! /usr/bin/env python3.12

import pandas as pd
import numpy as np
import cv2
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------- CONFIG ----------------
CSV_PATH = "dataset/labels.csv"
OUTPUT_H5 = "preprocessed_data_cameras.h5"
IMG_W, IMG_H = 200, 100
BATCH_CHUNK = 1
# ----------------------------------------

def load_and_preprocess(image_path, speed, lane_change, encoder):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Failed to load image {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Skipping image {image_path}: {str(e)}")
        return None, None, None

    speed = np.float32(float(speed) / 120.0)
    lane_change_encoded = encoder.transform([[lane_change]])[0].astype(np.float32)
    return img, speed, lane_change_encoded

def main():
    df = pd.read_csv(CSV_PATH)

    # First pass: filter valid rows
    valid_rows = []
    for _, row in df.iterrows():
        main_path = row["image"]
        right_path = row["image"].replace("_main", "_right")
        left_path = row["image"].replace("_main", "_left")

        #print(main_path + " " + right_path + " " + left_path)

        if (not os.path.exists(main_path) or
            not os.path.exists(right_path) or
            not os.path.exists(left_path)):
            continue
        valid_rows.append(row)

    filtered_df = pd.DataFrame(valid_rows)
    num_samples = len(filtered_df)

    # ---------- encoder ----------
    encoder = OneHotEncoder(
        categories=[[-1, 0, 1]],
        sparse_output=False,
        dtype=np.float32,
        handle_unknown="ignore"
    )
    encoder.fit([[-1], [0], [1]])

    # ---------- split indices ----------
    indices = np.arange(num_samples)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )

    # ---------- create HDF5 ----------
    with h5py.File(OUTPUT_H5, "w") as f:
        # Create datasets with correct size
        X_main_images = f.create_dataset(
            "X_main_images",
            shape=(num_samples, IMG_H, IMG_W, 3),
            dtype=np.float32,
            chunks=(BATCH_CHUNK, IMG_H, IMG_W, 3),
            compression="lzf",
        )

        X_right_images = f.create_dataset(
            "X_right_images",
            shape=(num_samples, IMG_H, IMG_W, 3),
            dtype=np.float32,
            chunks=(BATCH_CHUNK, IMG_H, IMG_W, 3),
            compression="lzf",
        )

        X_left_images = f.create_dataset(
            "X_left_images",
            shape=(num_samples, IMG_H, IMG_W, 3),
            dtype=np.float32,
            chunks=(BATCH_CHUNK, IMG_H, IMG_W, 3),
            compression="lzf",
        )

        X_speeds = f.create_dataset("X_speeds", (num_samples, 1), np.float32)
        X_lane_changes = f.create_dataset("X_lane_changes", (num_samples, 3), np.float32)
        y_steer = f.create_dataset("y_steer", (num_samples,), np.float32)
        y_throttle = f.create_dataset("y_throttle", (num_samples,), np.float32)
        y_brake = f.create_dataset("y_brake", (num_samples,), np.float32)

        # Process valid samples
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            main_img, speed, lane_change_encoded = load_and_preprocess(
                row["image"], row["speed"], row["lane_change"], encoder
            )

            right_img, _, _ = load_and_preprocess(
                row["image"].replace("_main", "_right"), row["speed"], row["lane_change"], encoder
            )

            left_img, _, _ = load_and_preprocess(
                row["image"].replace("_main", "_left"), row["speed"], row["lane_change"], encoder
            )

            X_main_images[i] = main_img
            X_right_images[i] = right_img
            X_left_images[i] = left_img
            X_speeds[i] = speed
            X_lane_changes[i] = lane_change_encoded
            y_steer[i] = row["steer"]
            y_throttle[i] = row["throttle"]
            y_brake[i] = row["brake"]

            print(f"Processed {i}/{num_samples}", end='\r')

        f.create_dataset("train_idx", data=train_idx)
        f.create_dataset("val_idx", data=val_idx)

    print("\nMulti-camera preprocessing complete ✔")
    filtered_df.to_csv("dataset/labels_filtered.csv", index=False)
    print(f"Saved filtered dataset with {num_samples} valid samples")

if __name__ == "__main__":
    main()

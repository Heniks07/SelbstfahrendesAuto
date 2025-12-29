#! /usr/bin/env python3.12

import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------- CONFIG ----------------
CSV_PATH = "dataset/labels.csv"
OUTPUT_H5 = "preprocessed_data_lidar.h5"
LIDAR_POINTS = 32  # Number of LIDAR points to use
# ----------------------------------------

def load_lidar_data(lidar_path):
    try:
        lidar_data = np.loadtxt(lidar_path, delimiter=',')
        # Use first LIDAR_POINTS points (closest points)
        return lidar_data[:LIDAR_POINTS, :3].astype(np.float32)
    except Exception as e:
        print(f"Error loading LIDAR data: {e}")
        return np.zeros((LIDAR_POINTS, 3), dtype=np.float32)

def main():
    df = pd.read_csv(CSV_PATH)
    num_samples = len(df)

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
        # LIDAR data
        X_lidar = f.create_dataset(
            "X_lidar",
            shape=(num_samples, LIDAR_POINTS, 3),
            dtype=np.float32,
        )

        # Other inputs
        X_speeds = f.create_dataset("X_speeds", (num_samples, 1), np.float32)
        X_lane_changes = f.create_dataset("X_lane_changes", (num_samples, 3), np.float32)

        # Outputs
        y_steer = f.create_dataset("y_steer", (num_samples,), np.float32)
        y_throttle = f.create_dataset("y_throttle", (num_samples,), np.float32)
        y_brake = f.create_dataset("y_brake", (num_samples,), np.float32)

        for i, row in df.iterrows():
            # Load LIDAR data
            lidar_path = f"dataset/lidar/{row['image'].replace("_main.png","").replace("dataset/images/","").split('_')[0]}.csv"
            lidar_data = load_lidar_data(lidar_path)

            # Process other inputs
            speed = np.float32(float(row["speed"]) / 120.0)
            lane_change_encoded = encoder.transform([[row["lane_change"]]])[0].astype(np.float32)

            X_lidar[i] = lidar_data
            X_speeds[i] = speed
            X_lane_changes[i] = lane_change_encoded
            y_steer[i] = row["steer"]
            y_throttle[i] = row["throttle"]
            y_brake[i] = row["brake"]

            print(f"Processed {i}/{num_samples}", end='\r')

        f.create_dataset("train_idx", data=train_idx)
        f.create_dataset("val_idx", data=val_idx)

    print("LIDAR preprocessing complete ✔")

if __name__ == "__main__":
    main()

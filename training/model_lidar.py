#! /usr/bin/env python3.12

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_lidar_model(input_lidar_shape=(32, 3), input_speed_shape=(1,), input_lane_shape=(3,)):
    # LIDAR input branch
    lidar_input = Input(shape=input_lidar_shape)
    x = Conv1D(32, 3, activation='relu')(lidar_input)
    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = Flatten()(x)
    lidar_branch = Dense(128, activation='relu')(x)

    # Speed input branch
    speed_input = Input(shape=input_speed_shape)
    speed_branch = Dense(16, activation='relu')(speed_input)

    # Lane change input branch
    lane_input = Input(shape=input_lane_shape)
    lane_branch = Dense(8, activation='relu')(lane_input)

    # Combine branches
    combined = Concatenate()([lidar_branch, speed_branch, lane_branch])
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)

    # Output layers
    steer_output = Dense(1, name='steer')(x)
    throttle_output = Dense(1, name='throttle')(x)
    brake_output = Dense(1, name='brake')(x)

    # Create model
    model = Model(
        inputs=[lidar_input, speed_input, lane_input],
        outputs=[steer_output, throttle_output, brake_output]
    )

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'steer': 'mse',
            'throttle': 'mse',
            'brake': 'mse',
        },
        metrics={
            'steer': 'mae',
            'throttle': 'mae',
            'brake': 'mae',
        }
    )
    return model

model = create_lidar_model()
model.summary()


import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

H5_PATH = "preprocessed_data_lidar.h5"
BATCH_SIZE = 16  # Even larger batch size
EPOCHS = 80

class HDF5LidarSequence(tf.keras.utils.Sequence):
    def __init__(self, h5_file, indices, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.f = h5_file
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_idx_sorted = np.sort(batch_idx)

        X_lidar = self.f["X_lidar"][batch_idx_sorted]
        X_spd = self.f["X_speeds"][batch_idx_sorted]
        X_lane = self.f["X_lane_changes"][batch_idx_sorted]

        y = {
            "steer": self.f["y_steer"][batch_idx_sorted],
            "throttle": self.f["y_throttle"][batch_idx_sorted],
            "brake": self.f["y_brake"][batch_idx_sorted],
        }

        order = np.argsort(np.argsort(batch_idx))
        X_lidar = X_lidar[order]
        X_spd = X_spd[order]
        X_lane = X_lane[order]
        for k in y:
            y[k] = y[k][order]

        return (X_lidar, X_spd, X_lane), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    f = h5py.File(H5_PATH, "r")

    train_seq = HDF5LidarSequence(f, f["train_idx"][:], BATCH_SIZE)
    val_seq = HDF5LidarSequence(f, f["val_idx"][:], BATCH_SIZE)

    checkpoint = ModelCheckpoint(
        "best_model_lidar.h5",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    plt.figure(figsize=(12, 8))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_history_lidar.png")

    f.close()

if __name__ == "__main__":
    main()

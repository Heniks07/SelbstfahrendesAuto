#! /usr/bin/env python3.12

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_multicam_model(input_image_shape=(100, 200, 3), input_speed_shape=(1,), input_lane_shape=(3,)):
    # Main camera branch
    main_input = Input(shape=input_image_shape)
    x1 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(main_input)
    x1 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x1)
    x1 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = Flatten()(x1)
    main_branch = Dense(100, activation='relu')(x1)

    # Right camera branch
    right_input = Input(shape=input_image_shape)
    x2 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(right_input)
    x2 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x2)
    x2 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = Flatten()(x2)
    right_branch = Dense(100, activation='relu')(x2)

    # Left camera branch
    left_input = Input(shape=input_image_shape)
    x3 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(left_input)
    x3 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x3)
    x3 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x3)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = Flatten()(x3)
    left_branch = Dense(100, activation='relu')(x3)

    # Speed input branch
    speed_input = Input(shape=input_speed_shape)
    speed_branch = Dense(16, activation='relu')(speed_input)

    # Lane change input branch
    lane_input = Input(shape=input_lane_shape)
    lane_branch = Dense(8, activation='relu')(lane_input)

    # Combine all branches
    combined = Concatenate()([main_branch, right_branch, left_branch, speed_branch, lane_branch])
    x = Dense(200, activation='relu')(combined)  # Larger dense layer for more inputs
    x = Dropout(0.3)(x)  # Slightly more dropout
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)

    # Output layers
    steer_output = Dense(1, name='steer')(x)
    throttle_output = Dense(1, name='throttle')(x)
    brake_output = Dense(1, name='brake')(x)

    # Create model
    model = Model(
        inputs=[main_input, right_input, left_input, speed_input, lane_input],
        outputs=[steer_output, throttle_output, brake_output]
    )

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Lower learning rate for more complex model
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

model = create_multicam_model()
model.summary()

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

H5_PATH = "preprocessed_data_cameras.h5"
BATCH_SIZE = 1
EPOCHS = 100

class HDF5MultiCamSequence(tf.keras.utils.Sequence):
    def __init__(self, h5_file, indices, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)  # This fixes the warning
        self.f = h5_file
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        # Get the batch indices and sort them
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_idx_sorted = np.sort(batch_idx)

        # Load data using sorted indices
        X_main = self.f["X_main_images"][batch_idx_sorted]
        X_right = self.f["X_right_images"][batch_idx_sorted]
        X_left = self.f["X_left_images"][batch_idx_sorted]
        X_spd = self.f["X_speeds"][batch_idx_sorted]
        X_lane = self.f["X_lane_changes"][batch_idx_sorted]

        y = {
            "steer": self.f["y_steer"][batch_idx_sorted],
            "throttle": self.f["y_throttle"][batch_idx_sorted],
            "brake": self.f["y_brake"][batch_idx_sorted],
        }

        # Return data in the correct order
        return (X_main, X_right, X_left, X_spd, X_lane), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    with h5py.File(H5_PATH, "r") as f:
        # Get the indices from the HDF5 file
        train_idx = f["train_idx"][:]
        val_idx = f["val_idx"][:]

        train_seq = HDF5MultiCamSequence(f, train_idx, BATCH_SIZE)
        val_seq = HDF5MultiCamSequence(f, val_idx, BATCH_SIZE)

        checkpoint = ModelCheckpoint(
            "best_model_cameras.h5",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=20,
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
        plt.savefig("training_history_cameras.png")

if __name__ == "__main__":
    main()

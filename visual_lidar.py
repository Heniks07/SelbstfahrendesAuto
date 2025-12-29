#! /usr/bin/env python3.12

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
import numpy as np

def create_lidar_model(input_lidar_shape=(32, 3), input_speed_shape=(1,), input_lane_shape=(3,)):
    # LIDAR input branch
    lidar_input = Input(shape=input_lidar_shape, name="lidar")
    x = Conv1D(32, 3, activation='relu')(lidar_input)
    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = Flatten()(x)
    lidar_branch = Dense(128, activation='relu')(x)

    # Speed input branch
    speed_input = Input(shape=input_speed_shape, name="speed")
    speed_branch = Dense(16, activation='relu')(speed_input)

    # Lane change input branch
    lane_input = Input(shape=input_lane_shape, name="lane change")
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

keras.utils.plot_model(model,to_file='lidar.png',show_layer_names=True)


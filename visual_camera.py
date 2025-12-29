#! /usr/bin/env python3.12

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
import numpy as np

def create_multicam_model(input_image_shape=(100, 200, 3), input_speed_shape=(1,), input_lane_shape=(3,)):
    # Main camera branch
    main_input = Input(shape=input_image_shape,name='Main Camera')
    x1 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(main_input)
    x1 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x1)
    x1 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = Flatten()(x1)
    main_branch = Dense(100, activation='relu')(x1)

    # Right camera branch
    right_input = Input(shape=input_image_shape, name='right camera')
    x2 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(right_input)
    x2 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x2)
    x2 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = Flatten()(x2)
    right_branch = Dense(100, activation='relu')(x2)

    # Left camera branch
    left_input = Input(shape=input_image_shape, name='left camera')
    x3 = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(left_input)
    x3 = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x3)
    x3 = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x3)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = Flatten()(x3)
    left_branch = Dense(100, activation='relu')(x3)

    # Speed input branch
    speed_input = Input(shape=input_speed_shape, name="speed")
    speed_branch = Dense(16, activation='relu')(speed_input)

    # Lane change input branch
    lane_input = Input(shape=input_lane_shape, name="lane change")
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

keras.utils.plot_model(model,to_file='camera.png',show_layer_names=True)


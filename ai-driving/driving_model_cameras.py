#! /usr/bin/env python3.12 

import numpy as np
import cv2
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

class DrivingModel:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.model.compile(
            optimizer='adam',
            loss={
                'steer': MeanSquaredError(),
                'throttle': MeanSquaredError(),
                'brake': MeanSquaredError(),
            },
            metrics={
                'steer': MeanAbsoluteError(),
                'throttle': MeanAbsoluteError(),
                'brake': MeanAbsoluteError(),
            }
        )
        self.right_camera = None
        self.left_camera = None

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if image is None:
            return np.zeros((100, 200, 3), dtype=np.float32)

        if hasattr(image, 'get_width'):  # Pygame surface
            img = pygame.surfarray.array3d(image).swapaxes(0, 1)
        else:  # Numpy array
            img = image
        img = cv2.resize(img, (200, 100))
        img = img / 255.0
        return img

    def set_camera_views(self, right_view, left_view):
        """Set the right and left camera views"""
        self.right_camera = right_view
        self.left_camera = left_view

    def predict(self, main_image, speed, lane_change):
        """Make predictions using all three camera views"""
        if main_image is None:
            return 0.0, 0.0, 0.0

        main_img = self.preprocess_image(main_image)
        right_img = self.preprocess_image(self.right_camera)
        left_img = self.preprocess_image(self.left_camera)

        main_img = np.expand_dims(main_img, axis=0)
        right_img = np.expand_dims(right_img, axis=0)
        left_img = np.expand_dims(left_img, axis=0)

        speed = np.array([[speed / 120.0]])
        lane_change_encoded = np.eye(3)[lane_change + 1].reshape(1, 3)

        steer, throttle, brake = self.model.predict(
            [main_img, right_img, left_img, speed, lane_change_encoded]
        )
        return steer[0][0], throttle[0][0], brake[0][0]

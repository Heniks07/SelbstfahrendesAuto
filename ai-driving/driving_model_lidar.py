#! /usr/bin/env python3.12 

import numpy as np
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
        self.lidar_data = None

    def set_lidar_data(self, lidar_data):
        """Set the LIDAR point cloud data"""
        try:
            # CARLA LIDAR data comes as a list of points
            points = []
            for detection in lidar_data:
                points.append([detection.point.x, detection.point.y, detection.point.z])
            points = np.array(points)
            self.lidar_data = points[:32, :3].reshape(1, 32, 3)
        except Exception as e:
            print(f"Error processing LIDAR data: {e}")
            self.lidar_data = None

    def predict(self, image, speed, lane_change):
        """Make predictions using LIDAR data"""
        if self.lidar_data is None:
            # Return neutral values if no LIDAR data
            return 0.0, 0.5, 0.0

        speed = np.array([[speed / 120.0]])
        lane_change_encoded = np.eye(3)[lane_change + 1].reshape(1, 3)

        steer, throttle, brake = self.model.predict(
            [self.lidar_data, speed, lane_change_encoded]
        )
        return steer[0][0], throttle[0][0], brake[0][0]

#! /usr/bin/env python3.12 

import numpy as np

class SpeedHold:
    def __init__(self):
        # These are instance variables, accessible with self.
        self.DEADZONE_KMH = 1.0
        self.SPEED_GAIN = 0.01
        self.OVERSPEED_MARGIN_KMH = 8.0
        self.BRAKE_GAIN = 0.02
        self.MAX_BRAKE = 0.5
        self.last_steer = 0.0

    def speed_hold(self, current_speed_kmh: float, target_speed: float = 90, base_throttle: float = 0.7):
        """
        Cruise-control style speed hold using throttle-first logic.
        """
        error = target_speed - current_speed_kmh

        throttle = base_throttle
        brake = 0.0

        # Deadzone to prevent micro-adjustments
        if abs(error) > self.DEADZONE_KMH:
            throttle = base_throttle + error * self.SPEED_GAIN

        throttle = np.clip(throttle, 0.0, 1.0)

        # Safety brake only if significantly overspeeding
        if current_speed_kmh > target_speed + self.OVERSPEED_MARGIN_KMH:
            brake = (current_speed_kmh - target_speed) * self.BRAKE_GAIN
            brake = np.clip(brake, 0.0, self.MAX_BRAKE)
            throttle = 0.0

        return throttle, brake

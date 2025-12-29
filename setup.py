#! /usr/bin/env python3.12
import carla

client = carla.Client('localhost', 2000)

world = client.load_world('town04')

# Slow-Mo for data collection
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

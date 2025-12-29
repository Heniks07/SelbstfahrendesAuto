#! /usr/bin/env python3.12

import carla
import numpy as np
import pygame
import os
import csv
from SpeedHold import SpeedHold
from datetime import datetime
import sys

# Setup CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()
location = world.get_map().get_spawn_points()[0].location
blueprint_library = world.get_blueprint_library()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.03
world.apply_settings(settings)
frame_id = 0

# Spawn vehicle
vehicle_bp = blueprint_library.find('vehicle.audi.a2')
spawn_point = carla.Transform(location, carla.Rotation())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Camera setup
def setup_camera(bp, position, rotation, width=640, height=480, fov=100):
    bp.set_attribute('image_size_x', str(width))
    bp.set_attribute('image_size_y', str(height))
    bp.set_attribute('fov', str(fov))
    return world.try_spawn_actor(
        bp,
        carla.Transform(position, rotation),
        attach_to=vehicle
    )

# Main camera (rear view)
main_camera = setup_camera(
    blueprint_library.find('sensor.camera.rgb'),
    carla.Location(x=0.5, z=1.5),
    carla.Rotation(),
    fov=100
)

# Right side camera
right_camera = setup_camera(
    blueprint_library.find('sensor.camera.rgb'),
    carla.Location(x=1.5, y=0.5, z=1.5),
    carla.Rotation(yaw=-45),
    width=320, height=240
)

# Left side camera
left_camera = setup_camera(
    blueprint_library.find('sensor.camera.rgb'),
    carla.Location(x=1.5, y=-0.5, z=1.5),
    carla.Rotation(yaw=45),
    width=320, height=240
)

# LIDAR sensor
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('range', '50')
lidar_bp.set_attribute('points_per_second', '10000')
lidar = world.try_spawn_actor(
    lidar_bp,
    carla.Transform(carla.Location(x=0, z=2.5), carla.Rotation()),
    attach_to=vehicle
)

# Initialize Pygame with multiple windows
pygame.init()
pygame.joystick.init()

# Create display arrangement
display_width = 640 + 320 + 320 + 40  # Main + right + left + spacing
display_height = max(480, 240) + 40   # Height + spacing

# Create a single large window with all views
main_display = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Multi-Camera View")

clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)

# Camera surfaces
surfaces = {
    'main': None,
    'right': None,
    'left': None
}

# Data collection
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/lidar", exist_ok=True)
csv_file = open("dataset/labels.csv", "a", newline="")
csv_writer = csv.writer(csv_file)

def process_camera_image(image, camera_name):
    global surfaces
    global frame_id
    frame_id = image.frame
    image_path = f"dataset/images/{datetime.today().strftime('%Y-%m-%d %H-%M-%S')} {frame_id:06d}_{camera_name}.png"
    image.save_to_disk(image_path)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]  # drop alpha
    array = array[:, :, ::-1]  # BGR → RGB
    surfaces[camera_name] = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def process_lidar_data(data):
    global frame_id
    # Convert LIDAR data to numpy array
    points = []
    for detection in data:
        points.append([detection.point.x, detection.point.y, detection.point.z])
    points = np.array(points)
    lidar_file = f"dataset/lidar/{datetime.today().strftime('%Y-%m-%d %H-%M-%S')} {frame_id:06d}.csv"
    np.savetxt(lidar_file, points, delimiter=',')


# Register callbacks
main_camera.listen(lambda image: process_camera_image(image, 'main'))
right_camera.listen(lambda image: process_camera_image(image, 'right'))
left_camera.listen(lambda image: process_camera_image(image, 'left'))
lidar.listen(process_lidar_data)

joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
if not joysticks:
    raise Exception("No controllers detected!")
controller = joysticks[0]

print('Entering main loop')

LINES = 6
def print_state(frame_id, circle, square, steer, throttle, brake):
    sys.stdout.write(f"{frame_id}\n")
    sys.stdout.write(f"{circle}\n")
    sys.stdout.write(f"{square}\n")
    sys.stdout.write(f"{steer}\n")
    sys.stdout.write(f"{throttle}\n")
    sys.stdout.write(f"{brake}\n")
    sys.stdout.flush()

steer = 0.0
throttle = 0.0
brake = 0.0
previous_pause_state = False
pause = False

print_state(frame_id, 0, 0, steer, throttle, brake)

try:
    while True:
        pygame.event.pump()

        speed = vehicle.get_velocity().length() * 3.6
        steer = round(controller.get_axis(0) * np.clip((20/np.clip(speed, 0.1, 250)), 0, 1), 2)
        manual_brake = (controller.get_axis(5) + 1)/2

        # Control logic
        speed_hold = SpeedHold()
        throttle, brake = speed_hold.speed_hold(speed)
        throttle = np.clip(throttle - manual_brake,0,1)
        brake = np.clip(brake + manual_brake - throttle, 0, 1)

        if brake > 0:
            throttle = 0

        square = controller.get_button(3)
        circle = controller.get_button(1)
        pause_state = controller.get_button(0)

        if pause_state != previous_pause_state and pause_state:
            pause = not pause
            print(f"Recording {'PAUSED' if pause else 'RESUMED'}")

        previous_pause_state = pause_state


        verbose = 1
        if(verbose):
            sys.stdout.write(f"\x1b[{LINES}A")
            for _ in range(LINES):
                sys.stdout.write("\x1b[2K\r\x1b[1B")
            sys.stdout.write(f"\x1b[{LINES}A")
            print_state(frame_id, circle, square, steer, throttle, brake)

        lane_change = -1 if square else (1 if circle else 0)

        # Update the single display with all cameras
        if surfaces['main'] is not None:
            main_display.fill((0, 0, 0))  # Clear display

            # Main camera (left)
            main_display.blit(surfaces['main'], (20, 20))

            # Right camera (middle)
            if surfaces['right'] is not None:
                main_display.blit(surfaces['right'], (660, 20))

            # Left camera (right)
            if surfaces['left'] is not None:
                main_display.blit(surfaces['left'], (980, 20))

            lane_change_label = ""
            if lane_change == -1:
                lane_change_label = "left"
            elif lane_change == 0:
                lane_change_label = "stay"
            else:
                lane_change_label = "right"

            # Speed display
            speed_text = font.render(f'Speed: {speed:.1f} km/h', True, (255, 255, 255))
            throttle_text = font.render(f'Throttle: {throttle*100:.0f}%', True, (255, 255, 255))
            brake_text = font.render(f'brake: {brake*100:.0f}%', True, (255, 255, 255))
            steer_text = font.render(f'Steer: {steer:.1f}', True, (255, 255, 255))
            lane_change_text = font.render(f'lane change: {lane_change_label}', True, (255, 255, 255))
            
            main_display.blit(speed_text, (250, 400))
            main_display.blit(throttle_text, (660, 300))
            main_display.blit(brake_text, (660, 330))
            main_display.blit(steer_text, (660, 360))
            main_display.blit(lane_change_text, (660, 390))


            # Camera labels
            main_label = font.render('MAIN', True, (255, 255, 255))
            right_label = font.render('RIGHT', True, (255, 255, 255))
            left_label = font.render('LEFT', True, (255, 255, 255))
            pause_text = font.render('PAUSED' if pause else 'RECORDING', True, (255, 0, 0) if pause else (0, 255, 0))

            main_display.blit(main_label, (20, 460))
            main_display.blit(right_label, (660, 240))
            main_display.blit(left_label, (980, 240))
            main_display.blit(pause_text, (660, 490))


        pygame.display.flip()
        clock.tick(15)

        if pause:
            continue

        # Write data to CSV
        csv_writer.writerow([
            frame_id,
            f"dataset/images/{datetime.today().strftime('%Y-%m-%d %H-%M-%S')} {frame_id:06d}_main.png",
            steer,
            throttle,
            brake,
            speed,
            lane_change
        ])

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        ))

        world.tick()


finally:
    main_camera.destroy()
    right_camera.destroy()
    left_camera.destroy()
    lidar.destroy()
    vehicle.destroy()
    print('exiting')

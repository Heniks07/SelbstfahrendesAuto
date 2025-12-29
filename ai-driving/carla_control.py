#! /usr/bin/env python3.12 

import carla
import numpy as np
import pygame
from SpeedHold import SpeedHold

def setup_sensors(world, vehicle, blueprint_library):
    """Setup all sensors and return them"""
    # Main camera (rear view)
    main_camera_bp = blueprint_library.find('sensor.camera.rgb')
    main_camera_bp.set_attribute('image_size_x', '640')
    main_camera_bp.set_attribute('image_size_y', '480')
    main_camera_bp.set_attribute('fov', '100')
    main_camera = world.try_spawn_actor(
        main_camera_bp,
        carla.Transform(carla.Location(x=0.5, z=1.5), carla.Rotation()),
        attach_to=vehicle
    )

    # Right camera
    right_camera_bp = blueprint_library.find('sensor.camera.rgb')
    right_camera_bp.set_attribute('image_size_x', '320')
    right_camera_bp.set_attribute('image_size_y', '240')
    right_camera_bp.set_attribute('fov', '100')
    right_camera = world.try_spawn_actor(
        right_camera_bp,
        carla.Transform(carla.Location(x=1.5, y=0.5, z=1.5), carla.Rotation(yaw=-45)),
        attach_to=vehicle
    )

    # Left camera
    left_camera_bp = blueprint_library.find('sensor.camera.rgb')
    left_camera_bp.set_attribute('image_size_x', '320')
    left_camera_bp.set_attribute('image_size_y', '240')
    left_camera_bp.set_attribute('fov', '100')
    left_camera = world.try_spawn_actor(
        left_camera_bp,
        carla.Transform(carla.Location(x=1.5, y=-0.5, z=1.5), carla.Rotation(yaw=45)),
        attach_to=vehicle
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

    return main_camera, right_camera, left_camera, lidar

def process_main_image(image):
    global main_surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    main_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def process_right_image(image):
    global right_surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    right_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def process_left_image(image):
    global left_surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    left_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    location = world.get_map().get_spawn_points()[0].location
    blueprint_library = world.get_blueprint_library()

    #Sync mode for pause
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # Spawn vehicle
    vehicle_bp = blueprint_library.find('vehicle.audi.a2')
    spawn_point = carla.Transform(location, carla.Rotation())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Setup sensors
    main_camera, right_camera, left_camera, lidar = setup_sensors(world, vehicle, blueprint_library)

    # Initialize surfaces
    global main_surface, right_surface, left_surface
    main_surface = None
    right_surface = None
    left_surface = None

    # Initialize control states
    manual_control = True
    paused = False
    lane_change = 0
    previous_pause_state = False
    previous_mode_state = False

    # Setup Pygame
    pygame.init()
    pygame.joystick.init()
    display_width = 640 + 320 
    display_height = max(480, 240)
    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA Multi-Camera View")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)

    # Register callbacks
    main_camera.listen(process_main_image)
    right_camera.listen(process_right_image)
    left_camera.listen(process_left_image)

    # Setup controller
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    if not joysticks:
        raise Exception("No controllers detected!")
    controller = joysticks[0]

    # Load both models
    try:
        from driving_model_cameras import DrivingModel as CameraModel
        camera_model = CameraModel('best_model_cameras.h5')
    except:
        print("Warning: Could not load camera model")
        camera_model = None

    try:
        from driving_model_lidar import DrivingModel as LidarModel
        lidar_model = LidarModel('best_model_lidar.h5')
        lidar.listen(lambda data: lidar_model.set_lidar_data(data))
    except:
        print("Warning: Could not load LIDAR model")
        lidar_model = None
    using_camera_model = True
    model = camera_model if camera_model is not None else lidar_model
    try:
        while True:
            pygame.event.pump()
            speed = vehicle.get_velocity().length() * 3.6

            # Get controller inputs
            pause_state = controller.get_button(9)
            mode_state = controller.get_button(2)
            sensor_state = controller.get_button(0)
            square = controller.get_button(3)
            circle = controller.get_button(1)
            steer_manual = round(controller.get_axis(0) * np.clip((20/np.clip(speed, 0.1, 250)), 0, 1), 2)
            manual_brake = (controller.get_axis(2) + 1) / 2

            # Handle pause state (edge detection)
            if pause_state and not previous_pause_state:
                paused = not paused
            previous_pause_state = pause_state

            # Handle control mode (edge detection)
            if mode_state and not previous_mode_state:
                manual_control = not manual_control
            previous_mode_state = mode_state

            # Handle sensor switch (edge detection)
            if sensor_state and not previous_sensor_state:
                if camera_model is not None and lidar_model is not None:
                    using_camera_model = not using_camera_model
                    model = camera_model if using_camera_model else lidar_model
            previous_sensor_state = sensor_state


            # Update display
            if main_surface is not None:
                display.fill((0, 0, 0))
                display.blit(main_surface, (0, 0))
                if right_surface is not None:
                    display.blit(right_surface, (640, 0))
                if left_surface is not None:
                    display.blit(left_surface, (640, 240))

                # Display info
                speed_text = font.render(f'Speed: {speed:.1f} km/h', True, (255, 255, 255))
                mode_text = font.render(f'Mode: {"MANUAL" if manual_control else "AI"}', True, (255, 255, 255))
                pause_text = font.render(f'PAUSED' if paused else '', True, (255, 0, 0))
                sensor_text = font.render(
                    f'Sensor: {"CAMERA" if using_camera_model else "LIDAR"}',
                    True,
                    (0, 255, 0) if using_camera_model else (255, 255, 0)
                )

                display.blit(speed_text, (0, 10))
                display.blit(mode_text, (0, 40))
                display.blit(pause_text, (0, 70))
                display.blit(sensor_text, (0, 100))

                # Camera labels
                main_label = font.render('MAIN', True, (255, 255, 255))
                right_label = font.render('RIGHT', True, (255, 255, 255))
                left_label = font.render('LEFT', True, (255, 255, 255))
                display.blit(main_label, (0, 460))
                display.blit(right_label, (650, 210))
                display.blit(left_label, (650, 450))

            pygame.display.flip()
            clock.tick(15)



            if paused:
                # No controls when paused
                continue

            # Handle lane change
            if square:
                lane_change = -1
            elif circle:
                lane_change = 1
            else:
                lane_change = 0

            if not manual_control and main_surface is not None:
                if using_camera_model and camera_model is not None:
                    camera_model.set_camera_views(right_surface, left_surface)
                    steer, throttle, brake = camera_model.predict(main_surface, speed, lane_change)
                elif lidar_model is not None:
                    steer, throttle, brake = lidar_model.predict(main_surface, speed, lane_change)
                else:
                    steer, throttle, brake = 0, 0, 0
            else:
                steer, throttle, brake = 0, 0, 0



            throttle, brake = SpeedHold().speed_hold(speed)

            # Apply controls
            if manual_control:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=float(np.clip(throttle, 0, 1)),
                    steer=float(np.clip(steer_manual, -1, 1)),
                    brake=float(np.clip(manual_brake + brake, 0, 1))
                ))
            else:
                vehicle.apply_control(carla.VehicleControl(
                    throttle=float(np.clip(throttle, 0, 1)),
                    steer=float(np.clip(steer, -1, 1)),
                    brake=float(np.clip(brake, 0, 1))
                ))

            world.tick()

    finally:
        main_camera.destroy()
        right_camera.destroy()
        left_camera.destroy()
        lidar.destroy()
        vehicle.destroy()

if __name__ == "__main__":
    main()

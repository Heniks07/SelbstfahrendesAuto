#! /usr/bin/env python3.12
import carla
import numpy as np
import time
import pygame
import os
import csv
from SpeedHold import SpeedHold
from datetime import datetime
import sys

#setup carla
client = carla.Client('localhost', 2000)
world = client.get_world()
location = world.get_map().get_spawn_points()[0].location
blueprint_library = world.get_blueprint_library()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.02  # <-- slows simulation
world.apply_settings(settings)

#spawn vehicle
vehicle_bp = blueprint_library.find('vehicle.audi.a2')
spawn_point = carla.Transform(location, carla.Rotation())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

#spawn camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')  # Width
camera_bp.set_attribute('image_size_y', '480')  # Height
camera_bp.set_attribute('fov', '100')           # Field of view


camera = world.try_spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=0.5, z=1.5), carla.Rotation()),  # Position behind the vehicle
    attach_to=vehicle
)

camera_right


pygame.init()
pygame.joystick.init() 
display = pygame.display.set_mode((640, 480))
pygame.display.set_caption("CARLA Camera View")
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)

surface = None
frame_id = 0
image_path = ""


#data collection
os.makedirs("dataset/images", exist_ok=True)
csv_file = open("dataset/labels.csv", "a", newline="")
print(csv_file)
csv_writer = csv.writer(csv_file)
#csv_writer.writerow(["frame", "image", "steer", "throttle", "brake", "speed", "lane_change"])



def process_camera_image(image):
    global surface
    global frame_id
    global image_path
    frame_id = image.frame
    image_path = f"dataset/images/{datetime.today().strftime('%Y-%m-%d %H-%M-%S')} {frame_id:06d}.png"
    image.save_to_disk(image_path)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]        # drop alpha
    array = array[:, :, ::-1]      # BGR → RGB

    # pygame expects (width, height)
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


LINES = 6
def print_state(frame_id, circle, square, steer, throttle, brake):
    sys.stdout.write(f"{frame_id}\n")
    sys.stdout.write(f"{circle}\n")
    sys.stdout.write(f"{square}\n")
    sys.stdout.write(f"{steer}\n")
    sys.stdout.write(f"{throttle}\n")
    sys.stdout.write(f"{brake}\n")
    sys.stdout.flush()



camera.listen(process_camera_image)

joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
if not joysticks:
    raise Exception("No controllers detected! Connect a game-controller to steer the car!")
controller = joysticks[0]



print('Entering main loop')

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



        steer = round(controller.get_axis(0) * np.clip((20/np.clip(speed,0.1,250)),0,1),2) #first adjust the sensitivity according to speed to make it easier to steer at high speed, then round to account for deadzone; 20 was chosing after some testing, 250 since I don't thin I can go faster or even get close
        #throttle = controller.get_axis(5) +1
        manual_brake = (controller.get_axis(2) +1)/2

        speed_hold = SpeedHold()

        throttle, brake = speed_hold.speed_hold(speed)

        brake = np.clip(brake+manual_brake,0,1)


        if brake > 0:
            throttle = 0


        square = controller.get_button(3)
        circle = controller.get_button(1)

        pause_state = controller.get_button(0)

        if(pause_state != previous_pause_state and pause_state):
           pause = not pause

        previous_pause_state = pause_state

        if(pause):
            continue

        if square:
            lane_change = -1  # left
        elif circle:
            lane_change = 1   # right
        else:
            lane_change = 0


        sys.stdout.write(f"\x1b[{LINES}A")

        # clear and rewrite
        for _ in range(LINES):
            sys.stdout.write("\x1b[2K\r\x1b[1B")

        # go back up again
        sys.stdout.write(f"\x1b[{LINES}A")

        print_state(frame_id, circle, square, steer, throttle, brake)

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        ))

        if(image_path != ""):
            csv_writer.writerow([
                frame_id,
                image_path,
                steer,
                throttle,
                brake,
                speed,
                lane_change
            ])


        
        world.tick()  # Update CARLA world

        if surface is not None:
            display.blit(surface, (0, 0))
            speed_text = font.render(f'Speed: {speed:.1f} km/h', True, (255, 255, 255))
            display.blit(speed_text, (10, 10))  # Position text in top-left corner


        pygame.display.flip()
        clock.tick(15)

finally:
    camera.destroy()
    vehicle.destroy()
    print('existing')

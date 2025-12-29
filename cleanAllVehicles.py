#!/usr/bin/env python3.12
import carla

# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

world.tick()
# Get all vehicles in the world
vehicles = world.get_actors().filter('vehicle.*')


# Destroy each vehicle
for vehicle in vehicles:
    print(f"Destroying vehicle: {vehicle.type_id}")
    vehicle.destroy()

world.tick()


print("All vehicles have been destroyed.")

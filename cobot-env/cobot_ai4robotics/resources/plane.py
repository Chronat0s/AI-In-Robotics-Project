import pybullet as p
import os
# This plane is here to stop everything from falling out of the world.

class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleplane.urdf')
        client.loadURDF(fileName=f_name,
                   basePosition=[0, 0, 0])



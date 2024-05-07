import pybullet as p
import os


class Ball:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), "model.urdf")
        self.id = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], base[2]])
        self.client = client

    # Have a lifetime timeout where it deletes itself after a number of steps.
    # https://stackoverflow.com/questions/3433559/python-time-delays
import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from cobot_ai4robotics.resources.car import Car # Have to swap these out.
from cobot_ai4robotics.resources.plane import Plane
from cobot_ai4robotics.resources.goal import Goal
import matplotlib.pyplot as plt
import time

import os
import pybullet_data

# Import objects from YCB. Remember to get training dataset from YCB for CNN.
from cobot_ai4robotics.resources.projectiles.ycb_objects.YcbBanana.banana import Banana
from cobot_ai4robotics.resources.projectiles.ycb_objects.YcbTennisBall.ball import Ball
from cobot_ai4robotics.resources.goal import Goal

# Import the robot object. Look into controlling it.
from cobot_ai4robotics.resources.cobot.kuka import Kuka
# from cobot_ai4robotics.resources.cobot.panda_env import pandaEnv

# Import the necessary library for YOLO.
from ultralytics import YOLO
from PIL import Image
from ultralytics.utils.plotting import Annotator

RENDER_HEIGHT = 640 # Increased resolution to help YOLO.
RENDER_WIDTH = 640

class CobotAI4RoboticsEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False, debug=0):
        if (isDiscrete):
            # self.action_space = gym.spaces.Discrete(15) # Direct motor control.
            self.action_space = gym.spaces.Discrete(7) # End effector xyz using IK.
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05], dtype=np.float32),
                high=np.array([.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            # [Current pose, projectile data (closest n projectiles?), xy distance to end effector goal (if doing task)],
            low=np.array([-40, -40], dtype=np.float32), # 0 - distance to nearest obstacle. 1 - distance to goal.
            high=np.array([40, 40], dtype=np.float32)) #
            # low=np.array([-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05, -1, -1, -1, -1, -1], dtype=np.float32), # -1 means no obstacle. If there is an obstacle, it is between 0 and the normalised upper limit.
            # high=np.array([.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05, 1, 1, 1, 1, 100], dtype=np.float32)) #
        self.np_random, _ = gym.utils.seeding.np_random()

        # Make the GUI client or not.
        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 1 # Time until next action.
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self._envStepCounter = 0

        # These variables are made for the AI4Robotics Final Project.
        self.cobot = None
        self.active_projectiles = []

        # Camera positioning and orientation.
        self._cam_dist = 6.5
        self._cam_yaw = 270
        self._cam_pitch = -5

        # Define which debug mode we are in.
        # 0 - run normally.
        # 1 - show yolo images.
        # 2 - dqn training.
        # 3 - run without projectiles and YOLO off..
        self.debug = debug

        # Load the YOLO model.
        if (self.debug != 2) & (self.debug != 3):
            self.YOLO = YOLO("./runs/detect/train4/weights/best.pt")
            print("YOLO LOADED")


        self.old_cobot_obs = 100

        # Init the Goals as none. Set them on reset or when goal achieved.
        self.goal = None

        self.reset()


    def step(self, action):
        ob = None # Placeholders until these functions get put in place.
        reward = 0

        # xyz, rpy
        # 15 actions, 7 up, 7 down, 1 no change.
        if self._isDiscrete:
            # actions = np.array([1,1,1,1,1,1,1, -1, -1, -1, -1, -1, -1, -1, 0])*0.5
            # motorIndex = [0,1,2,3,4,5,6,0,1,2,3,4,5,6,0]
            # motorIndex = motorIndex[action]
            # motorCommands = list(self.cobot.getPose())[0:7]
            # # print(motorCommands)
            # motorCommands[motorIndex] += actions[action]
            # print(len(motorCommands), motorCommands)
            # self.cobot.applyAction(motorCommands)

            dv = 0.5
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, 0, -dv, dv][action]
            # da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            # f = 0.3
            realAction = [dx, dy, dz, 0, 0]

            self.cobot.applyAction(realAction)

            # actions_d = [[0.1,0,0,0,0], # 0 - x up
            #        [0,0.1,0,0,0], # 1 - y up
            #        [0,0,0.1,0,0], # 2 - z up
            #        [-0.1,0,0,0,0], # 3 - x down
            #        [0,-0.1,0,0,0], # 4 - y down
            #        [0,0,-0.1,0,0], # 5 - z down
            #        [0,0,0,0,0]]#, # 6 - no move
            #     #    [0,0,0,0,np.pi/6], # 7 - rotate end effector +
            #     #    [0,0,0,0,-np.pi/6]] # 8 - rotate end effector -
            # self.cobot.applyAction(actions_d[action])
        else:
            self.cobot.applyAction(action)

        for i in range(self._actionRepeat):
            if ((self._envStepCounter % 80) == 0) & (self.debug != 3):
                self.generateProjectile()            
            self._p.stepSimulation()
            self.current_frame = self.refreshImage()
            # Observation space is [current pose, projectile yolo_pred data (closest n projectiles)]
            ob = self.getObservation()

            # Small reward for distance to goal. Positive if within 1m, negative if not.
            reward = 0.5 - ob[1]

            if ob[1] <= 0.15: # If reached goal.
                self.done = True
                reward = 100
            if ob[0] <= 0.5: # If near projectile.
                reward = -ob[0]*10

            # Check if the goal is reached and update the goal
            # end_effector_x = self.cobot.getPose()[0]  # Assuming this gets the x position of the end effector
            # goal_x = self.goal_positions[self.current_goal]

            # if (self.current_goal == 'left' and end_effector_x <= goal_x) or (self.current_goal == 'right' and end_effector_x >= goal_x):
            #     reward += 10  # Reward for reaching the goal
            #     self.current_goal = 'right' if self.current_goal == 'left' else 'left'  # Switch goal
            # Get points for holding the end effector up.

            # if self.cobot.getObservation()[2] > (self._h_table+0.3):
            #     reward += 1

            # Check for a hit and impose penalty (based on location?)
            for index, projectile in enumerate(self.active_projectiles):
                contact_points = p.getClosestPoints(self.cobot.kukaUid, projectile.id, distance = 0)
                if contact_points:
                    if self._renders:
                        print("Contact by ball no.", index, 'at point', f'({contact_points[0][5][0]:.2f}', f'{contact_points[0][5][1]:.2f}', f'{contact_points[0][5][2]:.2f})' , 'on KUKA.') 
                    # [0][5] - one point on the KUKA, [0][6] - one point on the banana

                    # End episode if hit.
                    self.done = True
                    reward = -500
                    break
                if self.done:
                    break
            # Check if hit table.
            contact_points = p.getClosestPoints(self.cobot.kukaUid, self.table_id, distance = 0)
            if contact_points:
                if self._renders:
                    print("Contact on table at point", f'({contact_points[0][5][0]:.2f}', f'{contact_points[0][5][1]:.2f}', f'{contact_points[0][5][2]:.2f})' , 'on KUKA.')
                self.done = True
                reward = -500

            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1        
        
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        # self._p.setGravity(0, 0, -9.81)
        Plane(self._p) # The floor. Stops everything from falling.
        self._envStepCounter = 0

        # Set up table
        self.table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                              basePosition=[0, 0.0, 0.0], useFixedBase=True)
        
        # Get the height of the table.
        table_info = p.getCollisionShapeData(self.table_id, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]/2

        # Set up cobot on top of table
        self.urdfRootPath=pybullet_data.getDataPath()
        self.cobot = Kuka(urdfRootPath=self.urdfRootPath, base_position=[0,0,self._h_table+0.08])
        # print(self.cobot._base_position)

        #Set a goal
        x = np.random.default_rng().uniform(0.3, 0.6, 1)[0]
        y_roll = np.random.default_rng().uniform(0, 1, 1)[0]
        if y_roll >= 0.5:
            y = np.random.default_rng().uniform(-0.5, -0.4, 1)[0]
        else:
            y = np.random.default_rng().uniform(0.4, 0.5, 1)[0]


        z = np.random.default_rng().uniform(self._h_table+0.2, self._h_table+0.6, 1)[0]     
        self.goal = np.array([x,y,z])

        self.goal_id = Goal(self._p, self.goal)

        # Set up camera
        self.render()
        self.current_frame = self.refreshImage()

        self.old_projectile = [-1,-1,-1,-1,-1]
        ob = self.getObservation()
        self.active_projectiles = []
        self.done = False

        self.old_cobot_obs = 100

        # Test obstacle generation
        # self.generateProjectile()

        return np.array(ob, dtype=np.float32)

    def _termination(self):
        return self._envStepCounter > 300

    def close(self):
        self._p.disconnect()

    # Below are functions made for the AI4Robotics Final Project.
    def generateProjectile(self):
        '''
            To be called as part of step() every n steps to make 1 new projectile.
        '''
        # Generate a single random projectile from a selection of URDFs
        # ~ 5m from the base plate of the cobot. Apply an external force
        # to it so that it moves towards an area randomly around the cobot,
        # but not at the immovable base plate section (that's just unfair!).

        # Random object spawn locations
        # dim | min | max |
        # X   | 6   | 9   |
        # Y   | -3  | 3   |
        # Z   | 0.1   | 3   |
        z_roll = np.random.default_rng().uniform(0, 1, 1)[0]
        if z_roll >= 0.5: # Low ball, on the sides.
            x = 4#np.random.default_rng().uniform(4.0, 9.0, 1)[0]
            y_roll = np.random.default_rng().uniform(0, 1, 1)[0]
            if y_roll >= 0.5:
                y = np.random.default_rng().uniform(0.36, 0.5, 1)[0]
            else:
                y = np.random.default_rng().uniform(-0.5, -0.36, 1)[0]
            z = np.random.default_rng().uniform(self._h_table+0.05, self._h_table+0.5, 1)[0]
        else: # High ball, free space.
            x = 4#np.random.default_rng().uniform(4.0, 9.0, 1)[0]
            y = np.random.default_rng().uniform(-0.5, 0.5, 1)[0]
            z = np.random.default_rng().uniform(self._h_table+0.5, self._h_table+0.7, 1)[0]          
        # x = 9
        # y = 0.35
        # z = self._h_table+0.7
        init_pos = np.array([x,y,z])

        # Allow no more than 15 active objects at a time.
        if len(self.active_projectiles) < 15:
            # self.obj = Banana(self._p, init_pos)
            self.obj = Ball(self._p, init_pos)
            self.active_projectiles.append(self.obj)
        else:
            # Get an object whose movement is finished. If there is none, skip this iteration and return.
            for index, projectile in enumerate(self.active_projectiles):
                pos, orn = self._p.getBasePositionAndOrientation(projectile.id)
                if (pos[0] < self.cobot.base_position[0] - 1) or (abs(pos[1]) > 2): # If passed cobot x position.
                    self.obj = self.active_projectiles[index]
                    self._p.resetBasePositionAndOrientation(self.obj.id, init_pos, orn)
                    break # only get one.
                elif index >= len(self.active_projectiles): 
                    return    


        # Targeting.
        mass = .058 # Mass of the ball.
        # y_target = np.random.default_rng().uniform(-0.5, 0.5, 1)[0]
        # z_target = np.random.default_rng().uniform(0, 2, 1)[0]
        # target = np.array(self.cobot.base_position) + np.array([0,y,z])#np.array([0, y, z])
        target = np.array([self.cobot.base_position[0],y,z])
        # Direction and distance to target.
        duration = 0.1 # Duration to apply force for. Larger is slower, smaller is faster. May miss target if too slow with gravity.

        # Initial velocity in 3d to hit target
        v0 = [(target[i] - init_pos[i]) / duration for i in range(3)]

        # Factor in gravity for initial velocity.
        # g = -9.81
        # v0[2] -= g * duration / 2

        # Required force calculation.
        force = [(v0[i] * mass) / duration for i in range(3)]

        self._p.applyExternalForce(objectUniqueId=self.obj.id, 
                             linkIndex=-1,
                             forceObj=force,
                             posObj =init_pos,
                             flags=p.WORLD_FRAME)
        
        # print("New obstacle with force: ", force, "at [", x, y, z, "]")

    def render(self, mode="rgb_array", close=False): # Add in an artificial camera on the table, facing the oncoming obstacles.
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self.cobot.kukaUid) # .kukaUid gives the base pos and orn.
        self.view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[base_pos[0]+5,0,self._h_table+0.5],
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        self.proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                        nearVal=0.1,
                                                        farVal=100)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                height=RENDER_HEIGHT,
                                                viewMatrix=self.view_matrix,
                                                projectionMatrix=self.proj_matrix,
                                                renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    def refreshImage(self):
        '''
        Call to refresh the RGB-D and segmentation images. These are returned by the function.
        '''
        return self._p.getCameraImage(RENDER_WIDTH, RENDER_HEIGHT, viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

    def getYOLOPrediction(self, show_pred=True):
        '''
        Pass the images from refreshImage to this function.
        The bounding boxes of the projectiles will be predicted using the RGB data.
        The depth value at the centre of the bounding box will be returned alongside
        the bounding box x min/max and y min/max values.
        '''
        # Reconstruct the rgb image as a PIL image to pass to the YOLO model.
        # https://stackoverflow.com/questions/70955660/how-to-get-depth-images-from-the-camera-in-pybullet
        rgb = np.reshape(self.current_frame[2], (RENDER_HEIGHT, RENDER_WIDTH, 4)).astype(np.uint8) # RGB-A. Comes as a list, turn into wxh array.
        # depth = np.reshape(self.current_frame[3], (RENDER_HEIGHT, RENDER_WIDTH, 1)) # Depth from 0.01 to 1 units.
        depth = np.array(self.current_frame[3])

        # print(rgb.shape, rgb)
        # print(depth.shape, depth)

        rgb_img = Image.fromarray(rgb[:,:,:3])

        pred = self.YOLO.predict(rgb_img, verbose=False)

        boxes = pred[0].boxes

        # Guide for converting rgbd to coordinates.
        # https://github.com/bulletphysics/bullet3/issues/1924 
        # Using the centre pixel, depth, and camera intrinsics, calculate the position of the obstacle in the world frame.
        # print("View Matrix: ", np.array(self.view_matrix).reshape(4,4))
        # print("Projection Matrix: ", np.array(self.proj_matrix).reshape(4,4))
        view_mat = np.asarray(self.view_matrix).reshape([4,4], order="F")
        proj_mat = np.asarray(self.proj_matrix).reshape([4,4], order="F")
        pixel_2_world_mat = np.linalg.inv(np.matmul(proj_mat, view_mat))
        # print(x_centre, y_centre, box_depth.item())

        y, x = np.mgrid[-1:1:2 / RENDER_HEIGHT, -1:1:2 / RENDER_WIDTH]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # Filter infinite depths.
        # pixels = pixels[z < 0.99] # May need to change upper limit?
        pixels[:,2] = 2 * pixels[:,2] - 1

        # Transform to world coordinates.
        coords = np.matmul(pixel_2_world_mat, pixels.T)
        coords /= coords[3:4]
        coords = coords[:3]
        coords = np.reshape(coords, (3, RENDER_HEIGHT, RENDER_WIDTH)) # Depth from 0.01 to 1 units.

        # print("coords:", coords.shape)

        # Get n boxes (normalised) and package them with the depth of the centroid.
        # https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
        
        if boxes:
            points = []
            ball_boxes = []
            for box in boxes:
                # Check box class and confidence.
                # print(int(box.cls), box.conf)
                if int(box.cls) != 39: # Tennis ball
                    continue
                if box.conf < 0.7:
                    continue

                b = box.xyxy[0]

                # Using b, calculate the centre pixel of the bounding box.
                x_centre = int((b[0] + b[2])/2)
                y_centre = int((b[1] + b[3])/2)

                # Access the depth image using the x_centre and y_centre pixel coordinates.
                # box_depth_n = depth[y_centre][x_centre] # Comes normalised between 0 and 1?
                # box_depth = 100 * 0.1 / (100 - (100 - 0.1) * box_depth_n)

                points.append(coords[:, y_centre, x_centre])
                
                # dist.append(box_depth)
                # ball_boxes.append(box)
                # print("Depth: ", box_depth_n, "at", x_centre, y_centre, "\n")

            # Zip the depth and boxes together.
            # ball_zip = zip(dist, ball_boxes)

            # # Sort the zip based on ascending depth.
            # sort_zip = sorted(ball_zip, key = lambda x : x[0])
            
            # Debugging. Shows the bounding boxes in a new window.
            if (self.debug == 1) & (self._envStepCounter % 20):
                annotator = Annotator(rgb_img)
                for box in ball_boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    annotator.box_label(b, self.YOLO.names[int(c)])
                annotator.show()

            # print(boxes.shape, boxes)
            # Flatten the array for insertion into observation.
            return points
        
        return False
        
    def getObservation(self):
        '''
        Build an observation of the current pose and projectiles.
        '''
        observation = []
        # Get current KUKA pose.
        # cobot_obs = self.cobot.getPose()[0:7] # Check if this is right.

        # Change from pose to end effector observations?
        # cobot_obs = self.cobot.getObservation()
        # print(cobot_obs)

        closest_projectile_data = [] # Placeholders for no obstacle detected.
        cobot_obs = self.old_cobot_obs
        yolo_pred = None
        projectilePos = None

        # Update YOLO predictions every n steps.
        if ((self._envStepCounter % 1) == 0):
            if ((self.debug == 2) | (self.debug == 3)): # If training the DQN, don't use YOLO as it is slow. Use simulation.
                if len(self.active_projectiles) == 0:
                    cobot_obs = 100 # No obstacle.
                # Get the current end effector position.
                kukaPos = np.array(self.cobot.getObservation()[0:3])
                min_dist = 100 # Sufficiently large number.
                for projectile in self.active_projectiles:
                    # For each projectile, get its position
                    projectilePos, _ = p.getBasePositionAndOrientation(projectile.id)
                    projectilePos = np.array(projectilePos)
                    # Get the distance to end effector.
                    dist = np.linalg.norm(projectilePos - kukaPos)
                    if dist < min_dist:
                        min_dist = dist
                        cobot_obs = dist

            # If not training the DQN.
            yolo_obs = self.old_cobot_obs
            if (self.debug != 2) & (self.debug != 3): # If not training, use YOLO.
                # yolo_pred should now be returning a list of obstacle positions. We must determine the closest one.
                yolo_pred = self.getYOLOPrediction()
                if yolo_pred:
                    # Calculate the closest distance to the end effector.
                    kukaPos = np.array(self.cobot.getObservation()[0:3])
                    min_dist = 100
                    for pos in yolo_pred:
                        dist = np.linalg.norm(pos - kukaPos)
                        if dist < min_dist:
                            min_dist = dist
                            yolo_obs = dist
                cobot_obs = yolo_obs

            # print(yolo_pred, projectilePos)
            # print(yolo_obs, cobot_obs)

        # Get the end effector distance to goal.
        goalDistance = np.linalg.norm(np.array(self.cobot.getObservation()[0:3]) - self.goal)

        observation.append(cobot_obs)
        observation.append(goalDistance)
        self.old_cobot_obs = cobot_obs
        # observation.extend(list(closest_projectile_data))
        # print("Observation: ", observation)
        return observation


import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import random
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import random
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import kuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class KukaGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=True,
               isDiscrete=True,
               maxSteps = 100,
               block_angle = 0):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._block_angle=block_angle

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self.seed()
    self.reset()
    self.observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * self.observationDim)
    if (self._isDiscrete):
      self.action_dim=4
      self.action_space = spaces.Discrete(self.action_dim)
    else:
       action_dim = 3
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):
    #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)

    self.xpos = 0.55 -0.07*random.random() + 0.07
    self.ypos = -0.1*random.random()+ 0.1
    # ang = 3.14*0.5+3.1415925438*random.random()
    # self.xpos= 0.55
    # self.ypos=0.08
    # self.ang=self._block_angle+3.14*random.random()
    self.ang=self._block_angle+3.14*random.random()
    orn = p.getQuaternionFromEuler([0,0,self.ang])
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), self.xpos,self.ypos,-0.15,orn[0],orn[1],orn[2],orn[3])

    p.setGravity(0,0,-10)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    for down in range(250):
      getdownAction=[0,0,-0.001]
      self._kuka.applyAction(getdownAction,terminate=False)
      p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    #  self._observation = self._kuka.getObservation()
     gripperState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
     gripperPos = gripperState[0]
     gripperOrn = gripperState[1]
     blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)

     invGripperPos,invGripperOrn = p.invertTransform(gripperPos,gripperOrn)
     gripperMat = p.getMatrixFromQuaternion(gripperOrn)
     dir0 = [gripperMat[0],gripperMat[3],gripperMat[6]]
     dir1 = [gripperMat[1],gripperMat[4],gripperMat[7]]
     dir2 = [gripperMat[2],gripperMat[5],gripperMat[8]]

     gripperEul =  p.getEulerFromQuaternion(gripperOrn)
     
     blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
     projectedBlockPos2D =[blockPosInGripper[0],blockPosInGripper[1]]
     blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
     

     #we return the relative x,y position and euler angle of block in gripper space
     blockInGripperPosXYEulZ =[blockPosInGripper[0],blockPosInGripper[1]]
     x_distance_state=self.xpos-self._kuka.endEffectorPos[0]
     y_distance_state=self.ypos-self._kuka.endEffectorPos[1]
     distance_state=[x_distance_state,y_distance_state]
    #  self._observation.extend(list(blockInGripperPosXYEulZ))
    #  self._observation.append(x_distance_state)
    #  self._observation.append(y_distance_state)
     self._observation=list(distance_state)
     return self._observation

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.01
      dx = [-dv,dv,0,0][action]
      dy = [0,0,-dv,dv][action]
      # f = -0.05
      realAction = [dx,dy,0,0.4,0]
      # print(realAction)
    else:
      #print("action[0]=", str(action[0]))
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      da = action[2] * 0.05
      f = 0.3
      realAction = [dx,dy,random.uniform(0,0.05),da,f]
    return self.step2( realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action,terminate=self._termination())
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()
    # npaction = np.array([action[3]]) #only penalize rotation until learning works well [action[0],action[1],action[3]])
    # actionCost = np.linalg.norm(npaction)*10.
    #print("actionCost")
    #print(actionCost)
    reward = self._reward()
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)


    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter>self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.22
    # closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid,maxDist)
    if self.xpos-0.01 < self._kuka.endEffectorPos[0] < self.xpos + 0.01:
            if self.ypos - 0.01 < self._kuka.endEffectorPos[1] < self.ypos + 0.01:
    # if (len(closestPoints)):#(actualEndEffectorPos[2] <= -0.43):
              self.terminated = 1
              fingerAngle = 0.4
      #print("terminating, closing gripper, attempting grasp")
              for down_2 in range(1000):
                getdownAction=[0,0,-0.001,fingerAngle,0]
                self._kuka.applyAction(getdownAction,terminate=True)
                closestPoints = p.getClosestPoints(self.blockUid,self._kuka.kukaUid,maxDist, -1, self._kuka.kukaEndEffectorIndex)
                p.stepSimulation()
                if (len(closestPoints)):
                  break
      #start grasp and terminate

              RotateAction = [0,0,0,fingerAngle,3.14-3.14*0.5-self.ang]
              self._kuka.applyAction(RotateAction,terminate=True)
              p.stepSimulation()
              for i in range (1000):
                graspAction = [0,0,0.000001,fingerAngle,0]
                self._kuka.applyAction(graspAction,terminate=True)
                p.stepSimulation()
                fingerAngle = fingerAngle-(0.4/1000.)
                if (fingerAngle<0):
                  fingerAngle=0

              for i in range (1000):
                graspAction = [0,0,0.001,fingerAngle,0]
                self._kuka.applyAction(graspAction,terminate=True)
                p.stepSimulation()
                blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
                if (blockPos[2] > 0.23):
                  #print("BLOCKPOS!")
                  #print(blockPos[2])
                  break
                state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2]>0.5):
                  break


              self._observation = self.getExtendedObservation()
              return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
    x_distance=abs(self.xpos-self._kuka.endEffectorPos[0])
    y_distance=abs(self.ypos-self._kuka.endEffectorPos[1])
    # closestPoints = p.getClosestPoints(self.blockUid,self._kuka.kukaUid,1000, -1, self._kuka.kukaEndEffectorIndex)
    total_distance=np.sqrt(x_distance**2+y_distance**2)
    reward = -total_distance
    # print('close:', closestPoints[0][8])
    # numPt = len(closestPoints)
    #print(numPt)
    # if (numPt>0):
    #   #print("reward:")
    #   reward = 1-closestPoints[0][8]
    if total_distance<0.02:
      reward = reward+1
    if (blockPos[2] >0.2):
      reward = reward+1
      # print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
     #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
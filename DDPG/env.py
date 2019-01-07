import pyglet
import pyglet
import numpy as np
from random import choice


class ArmEnv(object):
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal= {'x':320.,'y':120.,'l':40}
    goal_area={'x':325.,'y':325.,'l':150}
    state_dim = 9
    action_dim = 2


    def __init__(self):
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0
        self.right_place=0
        self.start_place=False

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)

        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done
    
    def step2(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal_area['x'] - a1xy_[0]) / 400, (self.goal_area['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal_area['x'] - finger[0]) / 400, (self.goal_area['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)

        # done and reward
        if self.goal_area['x'] - self.goal_area['l']/2 < finger[0] < self.goal_area['x'] + self.goal_area['l']/2:
            if self.goal_area['y'] - self.goal_area['l']/2 < finger[1] < self.goal_area['y'] + self.goal_area['l']/2:
                r += 1.
                self.right_place += 1
                if self.right_place > 50:
                    done = True
        else:
            self.right_place = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand()*300.
        self.goal['y'] = np.random.rand()*300.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal,self.start_place)
        self.viewer.start_place=self.start_place
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians

    def close(self):
        if(self.viewer):
           self.viewer.close()
           self.viewer = None

class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal,start_place):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.start_place=start_place
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        self.area=self.batch.add(
            3,pyglet.gl.GL_POLYGON,None,
            ('v2f',[250,400,
                    400,400,
                    400,250,]),
            ('c3B',(100, 100, 100) * 3,))
        if self.start_place==True:
            self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [xy21[0] - self.goal_info['l']/2, xy21[1] - self.goal_info['l']/2,
                    xy21[0] + self.goal_info['l']/2, xy21[1] - self.goal_info['l']/2,
                    xy21[0] + self.goal_info['l']/2, xy21[1] + self.goal_info['l']/2,
                    xy21[0] - self.goal_info['l']/2, xy21[1] + self.goal_info['l']/2]),
            ('c3B', (86, 109, 249) * 4))    # color
        else:
            self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        if self.start_place==True:
            self._update_place_arm()
        else:
            self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
         # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)
        
        # update arm
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
    

    def _update_place_arm(self):
         # update arm
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        # update goal
        self.goal.vertices = (
            xy21[0] - self.goal_info['l']/2, xy21[1] - self.goal_info['l']/2,
            xy21[0] + self.goal_info['l']/2, xy21[1] - self.goal_info['l']/2,
            xy21[0] + self.goal_info['l']/2, xy21[1] + self.goal_info['l']/2,
            xy21[0] - self.goal_info['l']/2, xy21[1] + self.goal_info['l']/2)

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y
    


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        print(env.start_place)
        env.start_place=True
        env.render()
        env.step(env.sample_action())
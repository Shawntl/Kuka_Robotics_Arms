from DDPG.env import ArmEnv
from DDPG.RL import DDPG
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 800
MAX_EPISODES2=500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
env2=ArmEnv2()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound


# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)
rl2 = rl


def train_first_net():
    # start training
    episode_reward=[0.0]
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                episode_reward[-1] += ep_r
                episode_reward.append(0.0)
                break
    rl.save()
    return episode_reward

def train_second_net():
    # start training
    converge_step_num=[0.0]
    for i in range(MAX_EPISODES2):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl2.choose_action(s)

            s_, r, done = env.step2(a)

            rl2.store_transition(s, a, r, s_)

            ep_r += r
            if rl2.memory_full:
                # start to learn once has fulfilled the memory
                rl2.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                converge_step_num[-1] += j
                converge_step_num.append(0.0)
                break
    rl2.save()
    return converge_step_num

def plotLearningCurve(env, MAX_EPISODES):
    episode_reward =train_first_net()
    rewards=episode_reward
    for episode_num in range(MAX_EPISODES-101):
        rewards[episode_num]=sum(rewards[episode_num:episode_num+100])/100
    rewards=rewards[0:701]
    plt.plot(np.arange(len(episode_reward)-100), rewards)
    plt.ylabel('rewards')
    plt.xlabel('episode')
    plt.show()
 

def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    rl2.restore()
    done=False
    while True:
        env.start_place=False
        s = env.reset()
        for _ in range(200):
            env.goal_second={'x':325.,'y':325.,'l':40}
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                print('done first stage')
                done=False
                while done==False:   
                    for _ in range(300):
                        env.start_place=True
                        env.render()
                        # print('next stage')
                        a=rl2.choose_action(s)
                        s,r,done=env.step2(a)
                        if done:
                            print('finish')
                            break
                       


if ON_TRAIN:
    plotLearningCurve(env, MAX_EPISODES) 
    train_second_net()
else:
    print('use_trained_layer')
    eval()


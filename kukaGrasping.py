import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
from KukaGymEnv import KukaGymEnv
import tensorflow as tf
from DQN_2 import DeepQNetwork


# from baselines import deepq


def train(env, num_episodes=1000):
    
    cost_his = []
    total_steps = 0
    episode_rewards = []
    episode_rewards = [0.0]
    reward=0
    record=0
    
    # act = deepq.load("kuka_model.pkl")
    # print(act)
    for episode in range(num_episodes):
        obs, done = env.reset(), False
        while not done:
            # env.render()
            action=RL.choose_action(obs)
            # print(action)
            obs_, reward, done, _ = env.step(action)
            step_reward=reward
            RL.store_transition(obs, action, step_reward, obs_)
            if (total_steps > 200):
                RL.learn()
            if done:
                print("num_epidodes:",episode,"episode_reward:",reward)
                break
            total_steps+=1
            obs=obs_
        episode_rewards[-1] += reward
        episode_rewards.append(0.0)
    print("training over")
    RL.sess.close()


if __name__ == '__main__':
    env= KukaGymEnv(renders=False, isDiscrete=True)
    RL=DeepQNetwork(env.action_dim, env.observationDim,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=500,
                      # output_graph=True
                      )
    train(env)
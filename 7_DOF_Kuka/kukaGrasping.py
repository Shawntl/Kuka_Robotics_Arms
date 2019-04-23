import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from KukaGymEnv import KukaGymEnv
import tensorflow as tf
from DDQN import DoubleDQN
from Prioritized_Replay_DQN import DQNPrioritizedReplay
from Dueling_DQN import DuelingDQN



MAX_EPISODES = 800
ON_TRAIN = True
plot_Q_eval=False
plotTrainHyper=False


with tf.variable_scope('Natural_DQN'):
    q_natural=DoubleDQN(
                        double_q=False,
                        n_actions=4, 
                        n_features=2,
                        learning_rate=0.001,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=1200,
                        memory_size=7000,
                        )

# with tf.variable_scope('Natural_DQN2'):
#     q_natural2=DoubleDQN(
#                         double_q=False,
#                         n_actions=4, 
#                         n_features=2,
#                         learning_rate=0.001,
#                         reward_decay=0.9,
#                         e_greedy=0.9,
#                         replace_target_iter=900,
#                         memory_size=7000,
#                         )

# with tf.variable_scope('Natural_DQN3'):
#     q_natural3=DoubleDQN(
#                         double_q=False,
#                         n_actions=4, 
#                         n_features=2,
#                         learning_rate=0.001,
#                         reward_decay=0.9,
#                         e_greedy=0.9,
#                         replace_target_iter=1200,
#                         memory_size=7000,
#                         )
# with tf.variable_scope('Natural_DQN4'):
#     q_natural4=DoubleDQN(
#                         double_q=False,
#                         n_actions=4, 
#                         n_features=2,
#                         learning_rate=0.001,
#                         reward_decay=0.9,
#                         e_greedy=0.9,
#                         replace_target_iter=2000,
#                         memory_size=7000,
#                         )
with tf.variable_scope('Double_DQN'):
    q_double=DoubleDQN(
                        double_q=True,
                        n_actions=4, 
                        n_features=2,
                        learning_rate=0.001,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=1200,
                        memory_size=7000,
                        # output_graph=True
                        )

with tf.variable_scope('DQN_with_prioritized_replay'):
    q_prio = DQNPrioritizedReplay(
                        n_actions=4, 
                        n_features=2, 
                        memory_size=7000,
                        e_greedy_increment=0.001, 
                        prioritized=True, 
                        )

with tf.variable_scope('dueling'):
    q_dueling = DuelingDQN(
                        n_actions=4, 
                        n_features=2, 
                        memory_size=7000,
                        e_greedy_increment=0.001, 
                        dueling=True,
                        )





def train(RL,env, num_episodes=MAX_EPISODES):
    
    cost_his = []
    total_steps = 0
    episode_rewards = [0.0]
    record=0
    
    # act = deepq.load("kuka_model.pkl")
    # print(act)
    for episode in range(num_episodes):
        obs, done = env.reset(), False
        while not done:
            # env.render()
            action=RL.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            step_reward=reward
            RL.store_transition(obs, action, step_reward, obs_)
            if (total_steps > 600):
                RL.learn()
            if done:
                # print("num_epidodes:",episode,"episode_reward:",reward)
                break
            total_steps+=1
            obs=obs_
        episode_rewards[-1] += reward
        episode_rewards.append(0.0)
    print("training over")
    # RL.save()
    RL.sess.close()
    return episode_rewards, RL.q

def plotLearningCurve(env, MAX_EPISODES,q_natural,q_double,q_prio,q_dueling):
    episode_rewards_dueling,q_dualing_value =train(q_dueling,env)
    episode_rewards_prio,q_prio_value =train(q_prio,env)
    episode_rewards_natural,q_natural_value =train(q_natural,env)
    episode_rewards_double,q_double_value =train(q_double,env)
    rewards_natural=episode_rewards_natural
    rewards_double=episode_rewards_double
    rewards_prio=episode_rewards_prio
    rewards_dueling=episode_rewards_dueling
    for episode_num_natural in range(MAX_EPISODES-101):
        rewards_natural[episode_num_natural]=sum(rewards_natural[episode_num_natural:episode_num_natural+100])/100
    rewards_natural=rewards_natural[0:699]
    for episode_num_double in range(MAX_EPISODES-101):
        rewards_double[episode_num_double]=sum(rewards_double[episode_num_double:episode_num_double+100])/100
    rewards_double=rewards_double[0:699]
    for episode_num_prio in range(MAX_EPISODES-101):
        rewards_prio[episode_num_prio]=sum(rewards_prio[episode_num_prio:episode_num_prio+100])/100
    rewards_prio=rewards_prio[0:699]
    for episode_num_dueling in range(MAX_EPISODES-101):
        rewards_dueling[episode_num_dueling]=sum(rewards_dueling[episode_num_dueling:episode_num_dueling+100])/100
    rewards_dueling=rewards_dueling[0:699]
    plt.plot(np.array(rewards_natural), c='r', label='natural')
    plt.plot(np.array(rewards_double), c='b', label='double')
    plt.plot(np.array(rewards_prio), c='g', label='prioritized')
    plt.plot(np.array(rewards_dueling), c='y', label='dueling')
    plt.ylabel('rewards')
    plt.xlabel('episode')
    plt.grid()
    plt.show()

def plotEvalQ_value(env,MAX_EPISODES,q_natural,q_double):
    episode_rewards_natural,q_natural_value =train(q_natural,env)
    episode_rewards_double,q_double_value =train(q_double,env)
    plt.plot(np.array(q_natural_value), c='r', label='natural')
    plt.plot(np.array(q_double_value), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()

def plotTrainHyper(env,MAX_EPISODES,q_natural1,q_natural2,q_natural3,q_natural4):
        episode_rewards_natural1,q_natural1_value =train(q_natural1,env)
        episode_rewards_natural2,q_natural2_value =train(q_natural2,env)
        episode_rewards_natural3,q_natural3_value =train(q_natural3,env)
        episode_rewards_natural4,q_natural4_value =train(q_natural4,env)
        rewards_natural1=episode_rewards_natural1
        rewards_natural2=episode_rewards_natural2
        rewards_natural3=episode_rewards_natural3
        rewards_natural4=episode_rewards_natural4
        rewards_natural=[rewards_natural1,rewards_natural2,rewards_natural3,rewards_natural4]
        for num_freq in range(4):
            for episode_num_natural in range(MAX_EPISODES-101):
                rewards_natural[num_freq][episode_num_natural]=sum(rewards_natural[num_freq][episode_num_natural:episode_num_natural+100])/100
            rewards_natural[num_freq]=rewards_natural[num_freq][0:699]
        

        plt.plot(np.array(rewards_natural[0]), c='r', label='update_freq=300')
        plt.plot(np.array(rewards_natural[1]), c='b', label='update_freq=900')
        plt.plot(np.array(rewards_natural[2]), c='g', label='update_freq=1200')
        plt.plot(np.array(rewards_natural[3]), c='y', label='update_freq=2000')
        plt.legend(loc='best')
        plt.ylabel('rewards')
        plt.xlabel('episodes')
        plt.grid()
        plt.show()





if __name__ == '__main__':
    if ON_TRAIN:
        env= KukaGymEnv(renders=False, isDiscrete=True)
        plotLearningCurve(env, MAX_EPISODES,q_natural,q_double,q_prio,q_dueling)
    if plot_Q_eval:
        env= KukaGymEnv(renders=False, isDiscrete=True)
        plotEvalQ_value(env,MAX_EPISODES,q_natural,q_double)
    if plotTrainHyper:
        env= KukaGymEnv(renders=False, isDiscrete=True)
        plotTrainHyper(env,MAX_EPISODES,q_natural1,q_natural2,q_natural3,q_natural4)
        
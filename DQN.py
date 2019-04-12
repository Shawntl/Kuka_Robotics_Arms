import os
import numpy as np
import numpy.random as npr
import random
import copy
from collections import defaultdict, namedtuple
from itertools import count
from more_itertools import windowed
from itertools import chain
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim





class LinearSchedule(object):                                                                                                                                                                                                                                           
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):                                                                                                                                                                                                     
        '''
        Linear interpolation between initial_p and final_p over                                                                                                                                                                                                      
        schedule_timesteps. After this many timesteps pass final_p is                                                                                                                                                                                                   
        returned.                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                        
        Args:                                                                                                                                                                                                                                                    
            - schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p                                                                                                                                                                                                                                                  
            - initial_p: initial output value                                                                                                                                                                                                                                        
            -final_p: final output value                                                                                                                                                                                                                                          
        '''                                                                                                                                                                                                                                                       
        self.schedule_timesteps = schedule_timesteps                                                                                                                                                                                                                    
        self.final_p = final_p                                                                                                                                                                                                                                          
        self.initial_p = initial_p                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                         
    def value(self, t):                                                                                                                                                                                                                                                 
        fraction = min(float(t) / self.schedule_timesteps, 1.0)                                                                                                                                                                                                         
        return self.initial_p + fraction * (self.final_p - self.initial_p)


Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state','done'))

class ReplayMemory(object):
    def __init__(self, size):
        '''
        Replay buffer used to store transition experiences. Experiences will be removed in a 
        FIFO manner after reaching maximum buffer size.
        
        Args:
            - size: Maximum size of the buffer.
        '''
        self.size = size
        self.memory = []
        self.idx = 0
        
    def add(self, *args):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.idx] = Transition(*args)
        self.idx = (self.idx + 1) % self.size
        self.memory_counter+=1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, state_shape, action_shape, lr=0.01):
        self.state_shape=state_shape
        self.action_shape=action_shape
        self.lr=lr
        
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        
        '''
        Deep Q-Network Tensorflow model.
        
        Args:
            - state_shape: Input state shape 
            - action_shape: Output action shape
        '''
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.state_shape], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.action_shape], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_shape, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.action_shape], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.action_shape], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.state_shape], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_shape, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.action_shape], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.action_shape], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
            
            
    def choose_action(self, observation,eps):
        observation=tf.convert_to_tensor(observation)
        observation=self.sess.run(observation)
        observation=np.array(observation)
        observation=observation.reshape(-1,9)
        if np.random.uniform() <eps:
            actions = np.random.randint(0, self.action_shape)
        else:
             # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            actions = np.argmax(actions_value)
        return actions

def optimize_model(session, policy_net, target_net, batch,batch_size, gamma):
    '''
    Calculates the target Q-values for the given batch and uses them to update the model.
    
    Args:
        - session: Tensorflow session
        - policy_net: Policy DQN model
        - target_net: DQN model used to generate target Q-values
        - batch: Batch of experiences uesd to optimize model
        - gamma: Discount factor
    '''
    batch=np.array(batch)
    batch=batch.reshape(-1,5)
    target_batch=batch[:,3]
    l1=[]
    for m in range(0,batch_size):
        for i in target_batch[m]:
            l1.append(i)
    target_batch=np.array(l1)
    target_batch=target_batch.reshape(-1,9)
    policy_batch=batch[:,0]
    l2=[]
    for n in range(0,batch_size):
        for j in policy_batch[n]:
            l2.append(j)
    policy_batch=np.array(l2)
    policy_batch=policy_batch.reshape(-1,9)
    
    target_net.q_next_batch, policy_net.q_eval_batch = session.run(
            [target_net.q_next, policy_net.q_eval],
            feed_dict={
                target_net.s_: target_batch,  # fixed params
                policy_net.s:  policy_batch,  # newest params
            })
    q_target = policy_net.q_eval_batch.copy()
    batch_index = np.arange(batch_size, dtype=np.int32)
    eval_act_index = batch[:, 1].astype(int)
    optimize_reward = batch[:, 2]
    q_target[batch_index, eval_act_index] =  optimize_reward + gamma * np.max(target_net.q_next_batch, axis=1)
    _, cost = session.run([policy_net._train_op, policy_net.loss],
                                             feed_dict={policy_net.s:  policy_batch,
                                             policy_net.q_target: q_target})
    return q_target,cost


def update_target_graph_op(tf_vars, tau):
    '''
    Creates a Tensorflow op which updates the target model towards the policy model by a small amount.
    
    Args:
        - tf_vars: All trainable variables in the Tensorflow graph
        - tau: Amount to update the target model
    '''
    total_vars = len(tf_vars)
    update_ops = list()
 
    for idx,var in enumerate(tf_vars[0:total_vars//2]):
        op = tf_vars[idx + total_vars//2].assign((var.value()*tau) + \
                                                 ((1-tau)*tf_vars[idx+total_vars//2].value()))
        update_ops.append(op)
    return update_ops

def update_target(session, update_ops):
    '''
    Calls each update op to update the target model.
    
    Args:
        - session: Tensorflow session
        - update_ops: The update ops which moves the target model towards the policy model
    '''
    for op in update_ops:
        session.run(op)


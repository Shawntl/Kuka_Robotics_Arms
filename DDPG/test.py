import gym
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

def main():
    
    env = KukaGymEnv(renders=True, isDiscrete=True)
    obs, done = env.reset(), False
    print("===================================")        
    print("obs")
    env.render()
           


if __name__ == '__main__':
    main()
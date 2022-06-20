import numpy as np
import gym
import turtlebot_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from fengs_algorithms.PPO.PPO import PPO

def learn():
    env_id = 'Turtlebot-v2'
    num_cpu = 6
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO(env, experiment_name='PPO', parallel=True)
    model.learn(10000000)

def test():
    env = gym.make('Turtlebot-v2', use_gui=True)
    model = PPO(env, experiment_name='PPO', parallel=False)
    model.load('./PPO/models/2022-06-20-20-48-05/6696960_PPO')
    model.test(100)

if __name__ == '__main__':
    test()




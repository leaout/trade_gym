import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_env.gym_env import TradingGymEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gym


def stock_trade(stock_file):
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: TradingGymEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    env = DummyVecEnv([lambda: TradingGymEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits

def train_test():
    df = pd.read_csv('./data/600519.txt',names=['Date','Open','High','Low','Close','Amount','Volume'],header=0)
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: TradingGymEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000)

    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    env.render()
    
def test_gym_env():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        # take a random action
        env.step(env.action_space.sample()) 
        # env.reset()
    # env.close()
if __name__ == '__main__':
    # test_gym_env()
    train_test()



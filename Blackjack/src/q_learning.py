#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
from q_learning_agent import QLearnAgent
import time


def train(gym_env, episode_count=10000):
    agent = QLearnAgent()
    agent.learn(env=gym_env, episode_count=episode_count)
    agent.show_reward_log(episode_count)
    return agent


def test(gym_env, agent, episode_count=10000):
    agent.learn(env=gym_env, episode_count=episode_count)
    agent.show_reward_log(episode_count)


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    start = time.time()
    # 1000000回で110sくらい。
    episode_count = 1000000
    training_agent = train(env, episode_count=episode_count)
    #test(env, training_agent, episode_count=episode_count)
    end = time.time()
    print('episode_count: {}, execute time: {}'.format(str(episode_count), str(end - start)))

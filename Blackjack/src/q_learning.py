#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import time
import matplotlib.pyplot as plt

from q_learning_agent import QLearnAgent


def train(gym_env, episode_count=10000):
    agent = QLearnAgent()
    agent.learn(env=gym_env, episode_count=episode_count)
    show_reward_log(agent.train_reward_log, episode_count, 'train_reward.png')
    show_reward_log(agent.test_reward_log, episode_count, 'test_reward.png')
    return agent


def test(gym_env, agent, episode_count=10000):
    agent.learn(env=gym_env, episode_count=episode_count)
    agent.show_reward_log(episode_count)


def show_reward_log(rewards, episode_count, path):
    interval = int(episode_count / 100)
    indices = list(range(0, len(rewards), interval))
    means = []
    stds = []
    for i in indices:
        rs = rewards[i:(i + interval)]
        means.append(np.mean(rs))
        stds.append(np.std(rs))
    means = np.array(means)
    stds = np.array(stds)
    plt.figure()
    plt.title("Reward History")
    plt.grid()
    plt.fill_between(indices, means - stds, means + stds,
                     alpha=0.1, color="g")
    plt.plot(indices, means, "o-", color="g",
             label="Rewards for each {} episode".format(interval))
    plt.legend(loc="best")
    plt.ylim([-0.25, 0.0])
    plt.savefig(path)


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    start = time.time()
    # 1000000回で110sくらい。
    episode_count = 1000000
    training_agent = train(env, episode_count=episode_count)
    #test(env, training_agent, episode_count=episode_count)
    end = time.time()
    print('episode_count: {}, execute time: {}'.format(str(episode_count), str(end - start)))

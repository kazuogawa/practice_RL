#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
from q_learning_agent import QLearnAgent


def train(gym_env):
    agent = QLearnAgent()
    agent.learn(env=gym_env, episode_count=10000)
    # agent.show_reward_log()
    # TODO: agentにQが溜まるのでどうにかみる
    return agent


def test(gym_env, agent):
    agent.learn(env=gym_env, episode_count=10000)
    agent.show_reward_log()


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    traning_agent = train(env)
    test(env, traning_agent)

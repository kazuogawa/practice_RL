#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
from q_learning_agent import QLearnAgent


def train():
    env = gym.make('Blackjack-v0')
    agent = QLearnAgent()
    agent.learn(env=env, episode_count=1000)
    agent.show_reward_log()
    # TODO: agentにQが溜まるのでどうにかみる


if __name__ == "__main__":
    train()

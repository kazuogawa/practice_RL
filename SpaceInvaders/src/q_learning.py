#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import gym


def train():
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
        env.render()


if __name__ == "__main__":
    train()
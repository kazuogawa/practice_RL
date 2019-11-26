import numpy as np
from enum import Enum


class ELAgent:
    def __init__(self, epsilon):
        # 各状態の各評価を持っている
        self.Q = {}
        self.epsilon = epsilon
        self.train_reward_log = []
        self.test_reward_log = []

    class Action(Enum):
        # カードを引かずに現在の手で勝負すること
        STAND = 0
        # カードをもう一枚引くこと
        HIT = 1

    def create_state(self, my_hand, dealer_hand, usable_ace):
        # 11以上でAceを持っていても1を11にするとoverしてしまい意味はないので、stateを共通化
        # return '{},{}'.format(my_hand, dealer_hand) if my_hand > 11 else '{},{},{}'.format(my_hand, dealer_hand, usable_ace)
        return '{},{},{}'.format(my_hand, dealer_hand, usable_ace)

    def epsilon_greedy_policy(self, state, actions, epsilon=None):
        # np.random.random()は標準のrandom.random()より早くて偏りがなく、推測しにくいらしい
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() >= epsilon:
            return actions[np.argmax(self.Q[state])]
        else:
            return np.random.choice(actions)

    def show_learning_log(self, episode):
        interval = 50
        rewards = self.train_reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
            episode, mean, std
        ))


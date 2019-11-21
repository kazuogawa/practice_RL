import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class ELAgent:
    def __init__(self, epsilon):
        # 各状態の各評価を持っている
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    class Action(Enum):
        # カードを引かずに現在の手で勝負すること
        STAND = 0
        # カードをもう一枚引くこと
        HIT = 1

    def create_state(self, my_hand, dealer_hand, usable_ace):
        # 11以上でAceを持っていても1を11にするとoverしてしまい意味はないので、stateを共通化
        # return '{},{}'.format(my_hand, dealer_hand) if my_hand > 11 else '{},{},{}'.format(my_hand, dealer_hand, usable_ace)
        return '{},{},{}'.format(my_hand, dealer_hand, usable_ace)

    def epsilon_greedy_policy(self, state, actions):
        # np.random.random()は標準のrandom.random()より早くて偏りがなく、推測しにくいらしい
        if np.random.random() >= self.epsilon and state in self.Q:
            return actions[np.argmax(self.Q[state])]
        else:
            return np.random.choice(actions)

    def init_log(self):
        self.reward_log = []

    def append_log(self, reward):
        self.reward_log.append(reward)

    def show_learning_log(self, episode):
        interval = 50
        rewards = self.reward_log[-interval:]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(
            episode, mean, std
        ))

    def show_reward_log(self, episode_count):
        interval = int(episode_count / 100)
        indices = list(range(0, len(self.reward_log), interval))
        means = []
        stds = []
        for i in indices:
            rewards = self.reward_log[i:(i + interval)]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
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
        plt.show()

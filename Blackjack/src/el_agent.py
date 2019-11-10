import numpy as np


class ELAgent:
    def __init__(self, epsilon):
        # 各状態の各評価を持っている
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    def epsilon_greedy_policy(self, state, actions):
        # np.random.random()は標準のrandom.random()より早くて偏りがなく、推測しにくいらしい
        if np.random.random() >= self.epsilon and state in self.Q and sum(self.Q[state]) != 0:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def append_log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self):
        # TODO:たまったreward_logを使ってうまく表示させる処理
        for log in self.reward_log:
            print(log)

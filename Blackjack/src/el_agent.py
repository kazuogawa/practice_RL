import numpy as np


class ELAgent:
    def __init__(self, epsilon):
        # 各状態の各評価を持っている
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []
        self.summary_reward = 0

    def create_state(self, my_hand, dealer_hand, usable_ace):
        # 11以上でAceを持っていても1を11にするとoverしてしまい意味はないので、stateを共通化
        # return '{},{}'.format(my_hand, dealer_hand) if my_hand > 11 else '{},{},{}'.format(my_hand, dealer_hand, usable_ace)
        return '{},{},{}'.format(my_hand, dealer_hand, usable_ace)

    def epsilon_greedy_policy(self, state, actions):
        # np.random.random()は標準のrandom.random()より早くて偏りがなく、推測しにくいらしい
        if np.random.random() >= self.epsilon and state in self.Q :
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def append_log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self):
        # TODO:たまったreward_logを使ってうまく表示させる処理
        print('sum_reward: {}'.format(self.summary_reward))

        print('my_hand,dealer_hand,usable_ace,no_draw_reward,draw_reward')
        for key, value in self.Q.items():
            print('{key},{value}'.format(key=str(key), value=str(value)))
        print('state log')
        for log in self.reward_log:
            print(log)

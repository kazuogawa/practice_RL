import numpy as np

from el_agent import ELAgent
from collections import defaultdict


class QLearnAgent(ELAgent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9, learning_rate=0.01):
        # ないと怒られるので書いておく・・・
        reward = 0
        # 取れるactionのkeyのlist.わかりやすくするためにEnumを使っている.list(range(env.action_space.n))と同じ
        actions = [self.Action.STAND, self.Action.HIT]
        self.Q = defaultdict(lambda: [0] * len(actions))
        for episode in range(episode_count):
            # (自分の手札の合計,相手の手札,1を11として使えるか否か)が返ってくる
            # https://github.com/openai/gym/blob/52e66f38081548e38711f51d4439d8bcc136d19e/gym/envs/toy_text/blackjack.py#L110
            my_hand, dealer_hand, usable_ace = env.reset()
            concat_state = self.create_state(my_hand, dealer_hand, usable_ace)
            done = False
            while not done:
                select_action = self.epsilon_greedy_policy(concat_state, actions).value

                next_state, reward, done, info = env.step(select_action)
                next_my_hand, next_dealer_hand, next_usable_ace = next_state[0], next_state[1], next_state[2]
                concat_next_state = self.create_state(next_my_hand, next_dealer_hand, next_usable_ace)
                # 勝負が終わっている時はconcat_next_stateは存在しないため、rewardのみをいれる
                gain = reward if done else reward + gamma * max(self.Q[concat_next_state])
                estimated = self.Q[concat_state][select_action]
                # V(s_t) ← V(s_t) + a(r_{t+1} + ¥gamma V(s_{t+1} - V(s_t))
                # TODO: s_{t+1}がs_Tのときはこれでいいのか？
                self.Q[concat_state][select_action] += learning_rate * (gain - estimated)
                concat_state = concat_next_state
            else:
                self.train_reward_log.append(reward)
            # test
            # TODO: 100 episodeごとにtestするのを試す
            # TODO: 上の学習するときの処理とほぼ同じなので、可能であればdefに切り抜いてまとめる
            my_hand, dealer_hand, usable_ace = env.reset()
            concat_state = self.create_state(my_hand, dealer_hand, usable_ace)
            while True:
                select_action = self.epsilon_greedy_policy(concat_state, actions, 0.0).value
                next_state, reward, done, info = env.step(select_action)
                next_my_hand, next_dealer_hand, next_usable_ace = next_state[0], next_state[1], next_state[2]
                concat_next_state = self.create_state(next_my_hand, next_dealer_hand, next_usable_ace)
                concat_state = concat_next_state
                if done:
                    self.test_reward_log.append(reward)
                    if episode % 1000 == 0:
                        print(episode, np.mean(self.test_reward_log))
                    break

from el_agent import ELAgent
from collections import defaultdict
import numpy as np
import tensorflow as tf
import keras


class QLearnAgent(ELAgent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        # ないと怒られるので書いておく・・・
        reward = 0
        self.init_log()
        # 取れるactionのkeyのlist. TODO:わかりにくいのでcase classのような物で代入したい
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        for episode in range(episode_count):
            # (自分の手札の合計,相手の手札,11を1として使えるか否か)が返ってくる
            # https://github.com/openai/gym/blob/52e66f38081548e38711f51d4439d8bcc136d19e/gym/envs/toy_text/blackjack.py#L110
            my_hand, dealer_hand, usable_ace = env.reset()
            concat_state = '{}:{}:{}'.format(my_hand, dealer_hand, usable_ace)
            done = False
            while not done:
                if render:
                    env.render()
                # 多分Q_learnなので学習ガッツリさせればこんな条件いらない・・・
                if my_hand == 21 or (my_hand == 10 and usable_ace):
                    select_action = 0
                else:
                    select_action = self.epsilon_greedy_policy(concat_state, actions)
                next_state, reward, done, info = env.step(select_action)
                next_my_hand, next_dealer_hand, next_usable_ace = next_state[0], next_state[1], next_state[2]
                concat_next_state = '{}:{}:{}'.format(next_my_hand, next_dealer_hand, next_usable_ace)
                # TODO: gainの計算これでいいのか？
                gain = reward + gamma * max(self.Q[concat_next_state])
                # TODO:epsilon greedyにいれかたを考える必要がある
                estimated = self.Q[concat_state][select_action]
                self.Q[concat_state][select_action] += learning_rate * (gain - estimated)
                concat_state = concat_next_state
            else:
                self.append_log('state:{state} -> reward: {reward}'.format(state=concat_state, reward=reward))
                self.summary_reward += reward

            if episode != 0 and episode % report_interval == 0:
                self.show_reward_log()

        actions = list(range(env.action_space.n))




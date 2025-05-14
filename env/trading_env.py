import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.balance = 1000
        self.crypto = 0
        self.last_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.crypto = 0
        self.last_price = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step][['open', 'high', 'low', 'close', 'volume']].values
        return np.append(obs, self.crypto)

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        if action == 1 and self.balance > price:
            self.crypto += 1
            self.balance -= price
        elif action == 2 and self.crypto > 0:
            self.crypto -= 1
            self.balance += price

        self.last_price = price
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self.balance + self.crypto * price - 1000
        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Crypto: {self.crypto}")

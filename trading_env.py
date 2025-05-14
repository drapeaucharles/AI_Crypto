import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, fee_percent=0.0004, one_trade_only=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.fee_percent = fee_percent
        self.one_trade_only = one_trade_only
        self.current_step = 0
        self.balance = 1000.0
        self.crypto = 0.0
        self.initial_balance = self.balance

        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(df.shape[1] - 1 + 1,),  # exclude timestamp, add position flag
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto = 0.0
        return self._get_obs()

    def _get_obs(self):
        obs = self.df.iloc[self.current_step, 1:].values.astype(np.float32)  # skip timestamp
        position = np.array([1.0 if self.crypto > 0 else 0.0], dtype=np.float32)
        return np.concatenate([obs, position])

    def step(self, action):
        done = False
        reward = 0.0

        if self.current_step >= len(self.df) - 1:
            done = True

        price = self.df.iloc[self.current_step]["close"]

        if action == 1 and self.crypto == 0:  # Buy
            self.crypto = self.balance / (price * (1 + self.fee_percent))
            self.balance = 0.0

        elif action == 2 and self.crypto > 0:  # Sell
            self.balance = self.crypto * price * (1 - self.fee_percent)
            self.crypto = 0.0

        self.current_step += 1

        net_worth = self.balance + self.crypto * price
        reward = net_worth - self.initial_balance
        self.initial_balance = net_worth

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        price = self.df.iloc[self.current_step]["close"]
        net_worth = self.balance + self.crypto * price
        print(f"Step: {self.current_step} | Price: {price:.2f} | Balance: {self.balance:.2f} | Crypto: {self.crypto:.4f} | Net Worth: {net_worth:.2f}")


import gym
import numpy as np
import pandas as pd

class AdvancedTradingEnv(gym.Env):
    def __init__(self, df_15m, df_1h, df_2h, df_4h, fee_percent=0.0004, leverage=5):
        super().__init__()
        self.df_15m = df_15m.reset_index(drop=True)
        self.df_1h = df_1h.reset_index(drop=True)
        self.df_2h = df_2h.reset_index(drop=True)
        self.df_4h = df_4h.reset_index(drop=True)
        self.fee_percent = fee_percent
        self.leverage = leverage

        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.crypto = 0.0
        self.current_step = 0
        self.position = None
        self.trades = []

        obs_len = self.df_15m.shape[1] - 1 + self.df_1h.shape[1] - 1 + self.df_2h.shape[1] - 1 + self.df_4h.shape[1] - 1 + 1
        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = long, 2 = short
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.crypto = 0.0
        self.current_step = 0
        self.position = None
        self.trades = []

        self.max_step = min(
            len(self.df_15m),
            len(self.df_1h),
            len(self.df_2h),
            len(self.df_4h)
        ) - 1

        return self._get_obs()

    def _get_obs(self):
        obs_15m = self.df_15m.iloc[self.current_step, 1:].values
        obs_1h = self.df_1h.iloc[self.current_step, 1:].values
        obs_2h = self.df_2h.iloc[self.current_step, 1:].values
        obs_4h = self.df_4h.iloc[self.current_step, 1:].values
        position_flag = np.array([1.0 if self.position else 0.0], dtype=np.float32)
        return np.concatenate([obs_15m, obs_1h, obs_2h, obs_4h, position_flag]).astype(np.float32)

    def step(self, action_info):
        done = False
        price = self.df_15m.iloc[self.current_step]["close"]
        reward = 0.0

        if isinstance(action_info, (tuple, list)):
            action = int(action_info[0])
            sl_pct = float(action_info[1])
            tp_pct = float(action_info[2])
        else:
            action = action_info
            sl_pct = 0.01
            tp_pct = 0.02

        if self.position:
            entry = self.position
            if entry["type"] == "long":
                if price <= entry["sl"]:
                    self._close_trade(price, "SL")
                elif price >= entry["tp"]:
                    self._close_trade(price, "TP")
            elif entry["type"] == "short":
                if price >= entry["sl"]:
                    self._close_trade(price, "SL")
                elif price <= entry["tp"]:
                    self._close_trade(price, "TP")

        if action in [1, 2] and not self.position:
            margin = self.balance * self.leverage
            self.crypto = margin / (price * (1 + self.fee_percent))
            sl_price, tp_price = price, price
            position_type = "long" if action == 1 else "short"

            if position_type == "long":
                sl_price = price * (1 - sl_pct)
                tp_price = price * (1 + tp_pct)
            else:  # short
                sl_price = price * (1 + sl_pct)
                tp_price = price * (1 - tp_pct)

            self.position = {
                "entry_price": price,
                "entry_step": self.current_step,
                "sl": sl_price,
                "tp": tp_price,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "type": position_type
            }
            self.balance = 0.0

        self.current_step += 1
        if self.current_step >= self.max_step:
            done = True

        net_worth = self.balance + (
            self.crypto * price if not self.position or self.position["type"] == "long"
            else self.crypto * (2 * self.position["entry_price"] - price)
        )
        reward = net_worth - self.initial_balance
        self.initial_balance = net_worth

        return self._get_obs(), reward, done, {"net_worth": net_worth}

    def _close_trade(self, price, reason):
        if self.position["type"] == "long":
            self.balance = self.crypto * price * (1 - self.fee_percent)
        elif self.position["type"] == "short":
            self.balance = (2 * self.position["entry_price"] - price) * self.crypto * (1 - self.fee_percent)

        trade = {
            "entry_price": self.position["entry_price"],
            "exit_price": price,
            "entry_step": self.position["entry_step"],
            "exit_step": self.current_step,
            "pnl": self.balance - self.initial_balance,
            "exit_reason": reason,
            "sl_pct": self.position["sl_pct"],
            "tp_pct": self.position["tp_pct"],
            "RR_ratio": self.position["tp_pct"] / self.position["sl_pct"] if self.position["sl_pct"] != 0 else None,
            "type": self.position["type"]
        }
        self.trades.append(trade)
        self.position = None
        self.crypto = 0.0

    def export_trades(self, path):
        pd.DataFrame(self.trades).to_csv(path, index=False)

    def render(self, mode="human"):
        price = self.df_15m.iloc[self.current_step]["close"]
        net_worth = self.balance + self.crypto * price if not self.position or self.position["type"] == "long" else self.crypto * (2 * self.position["entry_price"] - price)
        print(f"Step {self.current_step} | Price: {price:.2f} | Balance: {self.balance:.2f} | Crypto: {self.crypto:.4f} | Net Worth: {net_worth:.2f}")

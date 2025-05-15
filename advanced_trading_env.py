
import gym
import numpy as np
import pandas as pd

class AdvancedTradingEnv(gym.Env):
    def __init__(self, df_15m, df_1h, df_2h, df_4h, initial_balance=1000.0, max_positions=3):
        super().__init__()
        self.df_15m = df_15m.reset_index(drop=True)
        self.df_1h = df_1h.reset_index(drop=True)
        self.df_2h = df_2h.reset_index(drop=True)
        self.df_4h = df_4h.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.max_positions = max_positions
        self.positions = []
        self.trade_log = []
        self.current_step = 0
        self.sl_hits = 0
        self.cooldown = 0

        self.max_len = min(len(self.df_15m), len(self.df_1h), len(self.df_2h), len(self.df_4h))

        obs_space_size = self.df_15m.shape[1] - 1 + self.df_1h.shape[1] - 1 + self.df_2h.shape[1] - 1 + self.df_4h.shape[1] - 1
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.positions = []
        self.trade_log = []
        self.current_step = 100
        self.sl_hits = 0
        self.cooldown = 0

        if self.current_step >= self.max_len:
            raise ValueError("Not enough rows in one of the datasets to initialize the environment.")

        return self._get_obs()

    def _get_obs(self):
        obs_15m = self.df_15m.iloc[self.current_step, 1:].values.astype(np.float32)
        obs_1h = self.df_1h.iloc[self.current_step, 1:].values.astype(np.float32)
        obs_2h = self.df_2h.iloc[self.current_step, 1:].values.astype(np.float32)
        obs_4h = self.df_4h.iloc[self.current_step, 1:].values.astype(np.float32)
        return np.concatenate([obs_15m, obs_1h, obs_2h, obs_4h])

    def step(self, action):
        done = False
        reward = 0
        price = self.df_15m.iloc[self.current_step]["close"]

        open_trade = action[0] > 0.5
        sl_pct = np.clip(action[1], 0.001, 0.01)
        rr_ratio = np.clip(action[2] * 10, 2, 5)
        manual_exit = action[3] > 0.5
        close_index = int(np.clip(action[4] * self.max_positions, 0, self.max_positions - 1))

        if self.cooldown > 0:
            self.cooldown -= 1
        else:
            if open_trade and len(self.positions) < self.max_positions:
                risk_amount = self.balance * sl_pct
                sl_price = price - (price * sl_pct)
                tp_price = price + (price * sl_pct * rr_ratio)
                qty = risk_amount / (price * sl_pct)

                explanation = f"Opened trade at step {self.current_step} with SL {sl_pct*100:.2f}%, TP {rr_ratio:.2f}R (risk ${risk_amount:.2f}) based on decoded action."

                self.positions.append({
                    "entry_step": self.current_step,
                    "entry": price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "qty": qty,
                    "risk": risk_amount,
                    "action": action.tolist(),
                    "explanation": explanation
                })
                self.balance -= risk_amount

        new_positions = []
        for i, pos in enumerate(self.positions):
            sl, tp, entry, qty, risk = pos["sl"], pos["tp"], pos["entry"], pos["qty"], pos["risk"]
            result = ""
            exit_price = price
            pnl = 0

            if manual_exit and i == close_index:
                pnl = (price - entry) * qty
                self.balance += risk + pnl
                reward += pnl
                result = "manual_exit"
                explanation = f"Manually exited trade at step {self.current_step}. Price: {price:.2f}, PnL: {pnl:.2f}"
            elif price <= sl:
                self.sl_hits += 1
                self.cooldown = 96 if self.sl_hits >= 5 else 0
                reward -= risk
                result = "sl_hit"
                explanation = f"SL hit at step {self.current_step}. Price: {price:.2f}, Risk lost: {risk:.2f}"
            elif price >= tp:
                pnl = (tp - entry) * qty
                self.balance += risk + pnl
                reward += pnl
                result = "tp_hit"
                explanation = f"TP hit at step {self.current_step}. Price: {price:.2f}, PnL: {pnl:.2f}"
            else:
                new_positions.append(pos)
                continue

            self.trade_log.append({
                "step": self.current_step,
                "entry_step": pos["entry_step"],
                "entry_price": entry,
                "exit_price": exit_price,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "pnl": pnl,
                "result": result,
                "action": pos["action"],
                "explanation": pos["explanation"] + " | " + explanation
            })

        self.positions = new_positions
        self.current_step += 1
        if self.current_step >= self.max_len - 1:
            done = True

        net_worth = self.balance + sum([(price - p["entry"]) * p["qty"] + p["risk"] for p in self.positions])
        return self._get_obs(), reward, done, {"net_worth": net_worth}

    def render(self, mode="human"):
        price = self.df_15m.iloc[self.current_step]["close"]
        net_worth = self.balance + sum([(price - p["entry"]) * p["qty"] + p["risk"] for p in self.positions])
        print(f"Step {self.current_step} | Price: {price:.2f} | Balance: {self.balance:.2f} | Positions: {len(self.positions)} | Net Worth: {net_worth:.2f}")

    def export_trades(self, path="trades_log.csv"):
        df = pd.DataFrame(self.trade_log)
        df.to_csv(path, index=False)
        print(f"📄 Trades exported to {path}")

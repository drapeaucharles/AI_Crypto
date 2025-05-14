
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import pandas as pd

# Load datasets
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Load environment
env = AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)

# Load model
model = PPO.load("models/ppo_btc_advanced")

obs = env.reset()
done = False
net_worths = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    net_worths.append(info["net_worth"])

env.render()

# Plot net worth
import matplotlib.pyplot as plt
plt.plot(net_worths)
plt.title("Net Worth Over Time")
plt.xlabel("Steps")
plt.ylabel("Net Worth")
plt.grid(True)
plt.show()

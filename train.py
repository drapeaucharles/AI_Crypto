
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import pandas as pd
import os

os.makedirs("models", exist_ok=True)

# Load datasets
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Create environment
env = AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)

# Train model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs_advanced")
model.learn(total_timesteps=100_000_000)

# Save model
model.save("models/ppo_btc_advanced")
print("✅ Model trained and saved.")

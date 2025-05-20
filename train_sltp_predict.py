
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_output_policy import MultiOutputPolicy
from advanced_trading_env_sltp import AdvancedTradingEnv
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load feature data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Create environment factory
def make_env():
    def _init():
        return AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
    return _init

# ✅ Use a single environment instance for debugging
env = DummyVecEnv([make_env()])

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Initialize PPO model with custom policy
model = PPO(
    MultiOutputPolicy,
    env,
    verbose=1,
    device="cuda",
    n_steps=2048,
    batch_size=512,
    learning_rate=3e-4,
    tensorboard_log="./ppo_logs_sltp"
)

# Train for 5 million steps
print("__ Starting SL/TP Training with GPU Optimization...")
model.learn(total_timesteps=5_000_000)
model.save("models/ppo_sltp_final")
print("✅ Model saved to models/ppo_sltp_final.zip")

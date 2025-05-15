
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from advanced_trading_env import AdvancedTradingEnv
from multi_output_policy import MultiOutputPolicy
import pandas as pd
import os

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Create env
env = DummyVecEnv([lambda: AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)])

# Ensure models folder
os.makedirs("models", exist_ok=True)

# Train model
model = PPO(
    MultiOutputPolicy,
    env,
    verbose=1,
    device="cuda",
    n_steps=2048,
    batch_size=512,
    tensorboard_log="./ppo_logs_sltp"
)

print("🚀 Starting training with SL/TP prediction...")
model.learn(total_timesteps=1_000_000)
model.save("models/ppo_btc_sltp_predictor")
print("✅ Saved model with dynamic SL/TP prediction.")

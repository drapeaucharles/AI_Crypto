
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_output_policy import MultiOutputPolicy
from advanced_trading_env_sltp import AdvancedTradingEnv
import pandas as pd
import os

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Create a single environment
env = DummyVecEnv([lambda: AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)])

# Initialize PPO with safer parameters
model = PPO(
    MultiOutputPolicy,
    env,
    verbose=1,
    device="cuda",
    n_steps=512,
    batch_size=256,
    learning_rate=1e-5,
    tensorboard_log="./ppo_logs_sltp",
    target_kl=0.03
)

print("🚀 Starting training...")
model.learn(total_timesteps=5_000_000)
model.save("models/ppo_btc_sltp_final")
print("✅ Training complete and model saved.")


import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from multi_output_policy import MultiOutputPolicy
from advanced_trading_env_sltp import AdvancedTradingEnv

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

def make_env():
    def _init():
        return AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
    return _init

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)

    env = SubprocVecEnv([make_env() for _ in range(8)])

    os.makedirs("models", exist_ok=True)

    model = PPO(
        MultiOutputPolicy,
        env,
        verbose=1,
        device="cuda",
        n_steps=16384,
        batch_size=4096,
        learning_rate=3e-4,
        tensorboard_log="./ppo_logs_sltp"
    )

    print("🚀 Starting SL/TP Training with GPU Optimization...")
    model.learn(total_timesteps=5_000_000)
    model.save("models/ppo_btc_sltp_predictor")
    print("✅ Saved model to models/ppo_btc_sltp_predictor.zip")

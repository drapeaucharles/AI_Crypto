from stable_baselines3 import PPO
from env.trading_env import TradingEnv
import pandas as pd
import os

os.makedirs("models", exist_ok=True)

timeframes = {
    "15m": "data/btc_15m_features.csv",
    "1h": "data/btc_1h_features.csv",
    "2h": "data/btc_2h_features.csv",
    "4h": "data/btc_4h_features.csv"
}

for tf, path in timeframes.items():
    print(f"\n📊 Training PPO model for timeframe: {tf}")

    df = pd.read_csv(path).dropna().reset_index(drop=True)
    env = TradingEnv(df, fee_percent=0.0004)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./ppo_logs_{tf}")
    model.learn(total_timesteps=100_000)

    model_path = f"models/ppo_btc_{tf}.zip"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

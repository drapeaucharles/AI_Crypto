from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from utils.data_loader import load_data

if __name__ == "__main__":
    df = load_data("data/binance_data.csv")
    env = TradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/ppo_crypto")

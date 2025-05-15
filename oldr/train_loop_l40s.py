
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from advanced_trading_env import AdvancedTradingEnv
import pandas as pd
import os
import matplotlib.pyplot as plt

def make_env():
    def _init():
        df_15m = pd.read_csv("data/btc_15m_features.csv")
        df_1h = pd.read_csv("data/btc_1h_features.csv")
        df_2h = pd.read_csv("data/btc_2h_features.csv")
        df_4h = pd.read_csv("data/btc_4h_features.csv")
        return AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
    return _init

if __name__ == "__main__":
    # Load data once for evaluation env
    df_15m = pd.read_csv("data/btc_15m_features.csv")
    df_1h = pd.read_csv("data/btc_1h_features.csv")
    df_2h = pd.read_csv("data/btc_2h_features.csv")
    df_4h = pd.read_csv("data/btc_4h_features.csv")

    os.makedirs("models", exist_ok=True)

    # Create vectorized env
    env = SubprocVecEnv([make_env() for _ in range(4)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        n_steps=8192,
        batch_size=2048,
        learning_rate=3e-4,
        tensorboard_log="./ppo_logs_l40s"
    )

    net_worth_log = []

    for i in range(100):
        print(f"🚀 Block {i+1}/100 (Step {(i+1)*1_000_000:,})")
        model.learn(total_timesteps=1_000_000)
        checkpoint_path = f"models/ppo_btc_checkpoint_{i+1}.zip"
        model.save(checkpoint_path)
        print(f"✅ Saved checkpoint to {checkpoint_path}")

        # Evaluation with single non-vec env
        test_env = AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
        obs = test_env.reset()
        done = False
        net = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            net.append(info["net_worth"])

        final_net = net[-1]
        net_worth_log.append(final_net)
        print(f"📈 Checkpoint {i+1} Net Worth: ${final_net:.2f}")
        test_env.export_trades(f"models/trades_checkpoint_{i+1}.csv")

    plt.plot(net_worth_log, marker='o')
    plt.title("Final Net Worth After Each Checkpoint")
    plt.xlabel("Checkpoint")
    plt.ylabel("Net Worth")
    plt.grid(True)
    plt.savefig("models/training_progress_l40s.png")
    plt.show()

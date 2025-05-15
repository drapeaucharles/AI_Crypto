
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Create environment
env = AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)

# Initialize model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs_advanced")

# Net worth tracking
net_worth_log = []

# Train in 1M-step blocks for a total of 100M
for i in range(100):
    print(f"🚀 Starting block {i+1}/100 (step {(i+1)*1_000_000:,})")
    model.learn(total_timesteps=1_000_000)
    checkpoint_path = f"models/ppo_btc_checkpoint_{i+1}.zip"
    model.save(checkpoint_path)
    print(f"✅ Saved checkpoint to {checkpoint_path}")

    # Evaluate
    obs = env.reset()
    done = False
    net = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        net.append(info["net_worth"])

    final_net = net[-1]
    net_worth_log.append(final_net)
    print(f"📈 Checkpoint {i+1} Net Worth: ${final_net:.2f}")

    # Export trade log
    env.export_trades(f"models/trades_checkpoint_{i+1}.csv")

# Final plot
plt.plot(net_worth_log, marker='o')
plt.title("Final Net Worth After Each Checkpoint")
plt.xlabel("Checkpoint")
plt.ylabel("Net Worth")
plt.grid(True)
plt.savefig("models/training_progress.png")
plt.show()

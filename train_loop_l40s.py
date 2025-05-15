
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from advanced_trading_env import AdvancedTradingEnv
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Make environment factory
def make_env():
    def _init():
        return AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
    return _init

# Create parallel environments (adjust number for your CPU)
env = SubprocVecEnv([make_env() for _ in range(4)])  # 4 environments in parallel

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Initialize PPO model with GPU + large training parameters
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

# Net worth tracking
net_worth_log = []

# Train in 1M-step blocks
for i in range(100):  # Total: 100M steps
    print(f"🚀 Block {i+1}/100 (Step {(i+1)*1_000_000:,})")
    model.learn(total_timesteps=1_000_000)
    checkpoint_path = f"models/ppo_btc_checkpoint_{i+1}.zip"
    model.save(checkpoint_path)
    print(f"✅ Saved checkpoint to {checkpoint_path}")

    # Evaluate on a single instance for plotting
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

# Plot summary
plt.plot(net_worth_log, marker='o')
plt.title("Final Net Worth After Each Checkpoint")
plt.xlabel("Checkpoint")
plt.ylabel("Net Worth")
plt.grid(True)
plt.savefig("models/training_progress_l40s.png")
plt.show()

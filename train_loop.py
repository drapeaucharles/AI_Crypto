
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

# Train for 100 million timesteps
print("🚀 Starting 100M step training session...")
model.learn(total_timesteps=100_000_000)

# Save final model
model.save("models/ppo_btc_100M")
print("✅ Saved final model to models/ppo_btc_100M.zip")

# Evaluate after training
obs = env.reset()
done = False
net_worths = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    net_worths.append(info["net_worth"])

print(f"📈 Final Net Worth After 100M Training Steps: ${net_worths[-1]:.2f}")

# Plot net worth
plt.plot(net_worths)
plt.title("Net Worth Over Time After Training")
plt.xlabel("Steps")
plt.ylabel("Net Worth")
plt.grid(True)
plt.savefig("models/final_networth_plot.png")
plt.show()

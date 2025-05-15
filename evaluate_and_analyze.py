
import os
import glob
import pandas as pd
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import matplotlib.pyplot as plt

# Load the latest checkpoint
model_paths = sorted(glob.glob("models/ppo_btc_checkpoint_*.zip"))
if not model_paths:
    raise FileNotFoundError("❌ No checkpoint found in models/")
model_path = model_paths[-1]
print(f"📦 Evaluating checkpoint: {model_path}")

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# Create environment
env = AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
model = PPO.load(model_path)

obs = env.reset()
done = False
net_worths = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    net_worths.append(info["net_worth"])

# Export and analyze trades
trade_file = f"models/eval_trades_{os.path.basename(model_path).replace('.zip', '')}.csv"
env.export_trades(trade_file)
print(f"📝 Exported trades to {trade_file}")

df_trades = pd.read_csv(trade_file)
total_trades = len(df_trades)

if 'exit_reason' in df_trades.columns:
    sl_hits = df_trades[df_trades['exit_reason'] == 'SL']
    tp_hits = df_trades[df_trades['exit_reason'] == 'TP']
    manual_exits = df_trades[df_trades['exit_reason'] == 'MANUAL']
else:
    print("⚠️ 'exit_reason' not found in trades. Assuming all exits are manual or unknown.")
    sl_hits = tp_hits = []
    manual_exits = df_trades

# Final net worth
final_net = net_worths[-1]
initial_balance = 1000.0
profit_pct = (final_net - initial_balance) / initial_balance * 100

# Print summary
print("📊 Evaluation Summary:")
print(f"📈 Final Net Worth: ${final_net:.2f}")
print(f"💼 Total Trades: {total_trades}")
print(f"🏆 Win Rate: {len(df_trades[df_trades['pnl'] > 0]) / total_trades * 100:.2f}%")
print(f"💥 SL Hits: {len(sl_hits)} ({len(sl_hits)/total_trades*100:.1f}%)")
print(f"🎯 TP Hits: {len(tp_hits)} ({len(tp_hits)/total_trades*100:.1f}%)")
print(f"🛑 Manual Exits: {len(manual_exits)}")
print(f"📉 Max Drawdown: {min(df_trades['pnl'].cumsum()):.2f}")

# Optional: Plot equity curve
plt.plot(net_worths)
plt.title("Net Worth During Evaluation")
plt.xlabel("Steps")
plt.ylabel("Net Worth")
plt.grid(True)
plt.savefig("models/eval_networth_plot.png")
print("📊 Saved net worth plot to models/eval_networth_plot.png")

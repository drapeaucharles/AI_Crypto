
import os
import glob
import pandas as pd
import warnings
import logging
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv

# Suppress noisy SB3 warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Load data
df_15m = pd.read_csv("data/btc_15m_features.csv")
df_1h = pd.read_csv("data/btc_1h_features.csv")
df_2h = pd.read_csv("data/btc_2h_features.csv")
df_4h = pd.read_csv("data/btc_4h_features.csv")

# List all checkpoint models
model_paths = sorted(glob.glob("models/ppo_btc_checkpoint_*.zip"))
if not model_paths:
    raise FileNotFoundError("❌ No checkpoint files found.")

summary = []

for path in model_paths:
    checkpoint_name = os.path.basename(path).replace(".zip", "")
    print(f"📦 Evaluating: {checkpoint_name}")

    env = AdvancedTradingEnv(df_15m, df_1h, df_2h, df_4h)
    model = PPO.load(path, device='cpu')  # Force to CPU to suppress GPU warning

    obs = env.reset()
    done = False
    net_worths = []

    while not done:
        action, *_ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        net_worths.append(info["net_worth"])

    final_net = net_worths[-1]
    trade_file = f"models/eval_trades_{checkpoint_name}.csv"
    env.export_trades(trade_file)

    df_trades = pd.read_csv(trade_file)
    total_trades = len(df_trades)
    win_rate = (df_trades['pnl'] > 0).mean() * 100 if total_trades > 0 else 0
    max_drawdown = df_trades['pnl'].cumsum().min() if total_trades > 0 else 0

    if 'exit_reason' in df_trades.columns:
        sl_count = (df_trades['exit_reason'] == 'SL').sum()
        tp_count = (df_trades['exit_reason'] == 'TP').sum()
        manual_count = (df_trades['exit_reason'] == 'MANUAL').sum()
    else:
        sl_count = tp_count = 0
        manual_count = total_trades

    summary.append({
        "Checkpoint": checkpoint_name,
        "NetWorth": round(final_net, 2),
        "Trades": total_trades,
        "WinRate %": round(win_rate, 2),
        "SL": sl_count,
        "TP": tp_count,
        "Manual": manual_count,
        "Drawdown": round(max_drawdown, 2)
    })

# Display clean table
df_summary = pd.DataFrame(summary)
df_summary.sort_values(by="Checkpoint", inplace=True)
df_summary.reset_index(drop=True, inplace=True)

print("\n📊 Evaluation Summary (all checkpoints):\n")
print(df_summary.to_string(index=False))
df_summary.to_csv("models/eval_summary.csv", index=False)
print("\n✅ Summary saved to models/eval_summary.csv")

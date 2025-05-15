
import os
import pandas as pd

def analyze_trades(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        return None

    total = len(df)
    tp = df[df['exit_reason'] == 'TP'].shape[0]
    sl = df[df['exit_reason'] == 'SL'].shape[0]
    manual = df[df['exit_reason'] == 'manual'].shape[0] if 'manual' in df['exit_reason'].values else 0
    winrate = (tp / total) * 100 if total > 0 else 0
    avg_rr = df['RR_ratio'].mean() if 'RR_ratio' in df else None
    net_pnl = df['pnl'].sum()
    max_drawdown = df['pnl'].cumsum().min()

    return {
        "File": os.path.basename(file_path),
        "Trades": total,
        "WinRate %": round(winrate, 2),
        "TP": tp,
        "SL": sl,
        "Manual": manual,
        "Avg RR": round(avg_rr, 2) if avg_rr else None,
        "Net PnL": round(net_pnl, 2),
        "Drawdown": round(max_drawdown, 2)
    }

def main():
    trade_dir = "models"
    results = []
    for file in sorted(os.listdir(trade_dir)):
        if file.startswith("trades_checkpoint_") and file.endswith(".csv"):
            path = os.path.join(trade_dir, file)
            summary = analyze_trades(path)
            if summary:
                results.append(summary)

    df_result = pd.DataFrame(results)
    print(df_result.to_string(index=False))

if __name__ == "__main__":
    main()

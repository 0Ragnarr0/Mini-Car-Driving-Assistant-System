"""Live training-progress viewer — watch the car learn in real time.

Run this in a SECOND terminal while train_metadrive.py is training. It reads the
metrics CSV and refreshes a plot every couple seconds (return + arrive/crash rates),
so you see the learning curve climb like watching an ML-Agents run.

Usage:
    python plot_progress.py                # auto-picks the newest training CSV
    python plot_progress.py path/to.csv    # a specific run
"""

import sys
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_CONFIG


def newest_csv():
    logs = DATA_CONFIG.get("logs_dir", "logs")
    files = glob.glob(os.path.join(logs, "train_metadrive_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


def read_csv(path):
    eps, rets, avg, succ, col = [], [], [], [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                eps.append(int(p[0])); rets.append(float(p[2])); avg.append(float(p[3]))
                succ.append(int(p[4])); col.append(int(p[5]))
    except FileNotFoundError:
        pass
    return eps, rets, avg, succ, col


def rolling(x, window=20):
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - window + 1): i + 1].mean() for i in range(len(x))])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest_csv()
    if not path:
        print("No training CSV found yet. Start train_metadrive.py first.")
        return
    print(f"[Viewer] Watching {path}  (Ctrl+C to stop)")

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    while True:
        eps, rets, avg, succ, col = read_csv(path)
        if eps:
            ax1.clear(); ax2.clear()
            ax1.plot(eps, rets, color="#9bbcff", alpha=0.5, label="episode return")
            ax1.plot(eps, avg, color="#1f6feb", linewidth=2, label="avg(10)")
            ax1.set_ylabel("return"); ax1.grid(alpha=0.3); ax1.legend(loc="upper left")
            ax1.set_title(f"MetaDrive TD3 progress — {len(eps)} episodes")
            ax2.plot(eps, rolling(succ), color="#2ea043", linewidth=2, label="arrive rate (20)")
            ax2.plot(eps, rolling(col), color="#d9534f", linewidth=2, label="crash rate (20)")
            ax2.set_ylabel("rate"); ax2.set_xlabel("episode"); ax2.set_ylim(-0.05, 1.05)
            ax2.grid(alpha=0.3); ax2.legend(loc="upper left")
            fig.tight_layout()
        plt.pause(2.0)  # refresh every 2 s


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Viewer] stopped.")

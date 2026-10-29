"""Бърза визуализация на drift в генерираните datasets."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

SYNTHETIC = Path("data/synthetic")
FIG_DIR = Path("notebooks/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

datasets = ["stable", "sudden", "gradual", "recurring"]

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

for ax, dtype in zip(axes, datasets):
    path = SYNTHETIC / f"consumption_{dtype}.parquet"
    df = pd.read_parquet(path)
    daily = df["consumption_mw"].resample("D").mean()
    ax.plot(daily.index, daily.values, linewidth=0.8, color="#2c3e50")
    ax.set_title(f"{dtype.upper()}", fontweight="bold", loc="left")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.3)

plt.tight_layout()
out = FIG_DIR / "00_drift_check.png"
plt.savefig(out, dpi=100)
print(f"Saved: {out}")

# Числова проверка: first half vs second half mean
print()
print("=" * 70)
print(f"{'Dataset':<12} | {'1st half':>10} | {'2nd half':>10} | {'Δ %':>8}")
print("=" * 70)
for dtype in datasets:
    df = pd.read_parquet(SYNTHETIC / f"consumption_{dtype}.parquet")
    n = len(df)
    first = df["consumption_mw"].iloc[: n // 2].mean()
    second = df["consumption_mw"].iloc[n // 2 :].mean()
    delta = (second - first) / first * 100
    print(f"{dtype:<12} | {first:>10.1f} | {second:>10.1f} | {delta:>+7.2f}%")
print("=" * 70)
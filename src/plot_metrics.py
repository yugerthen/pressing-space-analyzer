import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/metrics_timeseries.csv")
df["timestamp_min"] = df["timestamp_s"] / 60
df = df.sort_values("timestamp_min").reset_index(drop=True)

window = 3
df["teamA_hull_smooth"] = df["teamA_hull_area"].rolling(window, center=True, min_periods=1).mean()
df["teamB_hull_smooth"] = df["teamB_hull_area"].rolling(window, center=True, min_periods=1).mean()
df["teamA_press_smooth"] = df["teamA_avg_nearest_opp"].rolling(window, center=True, min_periods=1).mean()
df["teamB_press_smooth"] = df["teamB_avg_nearest_opp"].rolling(window, center=True, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(df["timestamp_min"], df["teamA_hull_area"], "o", color="red", alpha=0.25)
axes[0].plot(df["timestamp_min"], df["teamB_hull_area"], "o", color="blue", alpha=0.25)
axes[0].plot(df["timestamp_min"], df["teamA_hull_smooth"], "-", color="red", linewidth=2.5, label="Équipe A (lissé)")
axes[0].plot(df["timestamp_min"], df["teamB_hull_smooth"], "-", color="blue", linewidth=2.5, label="Équipe B (lissé)")
axes[0].set_ylabel("Aire du hull (px²)")
axes[0].set_title("Occupation spatiale des équipes dans le temps (points bruts + tendance lissée)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df["timestamp_min"], df["teamA_avg_nearest_opp"], "o", color="red", alpha=0.25)
axes[1].plot(df["timestamp_min"], df["teamB_avg_nearest_opp"], "o", color="blue", alpha=0.25)
axes[1].plot(df["timestamp_min"], df["teamA_press_smooth"], "-", color="red", linewidth=2.5, label="Équipe A (lissé)")
axes[1].plot(df["timestamp_min"], df["teamB_press_smooth"], "-", color="blue", linewidth=2.5, label="Équipe B (lissé)")
axes[1].set_ylabel("Distance moy. adversaire proche (px)")
axes[1].set_xlabel("Minute du match")
axes[1].set_title("Pressing subi dans le temps (points bruts + tendance lissée)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../data/tactical_timeseries.png", dpi=150)
print("Graphique sauvegardé")
import csv

with open("../data/metrics_timeseries.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

clean_rows = [
    r for r in rows
    if r["teamA_hull_area"] not in ("", "None", None)
    and r["teamB_hull_area"] not in ("", "None", None)
]

print(f"{len(rows)} lignes au total, {len(clean_rows)} lignes exploitables après nettoyage")

with open("../data/metrics_timeseries_clean.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(clean_rows)

print("Sauvegardé dans data/metrics_timeseries_clean.csv")
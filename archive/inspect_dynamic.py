
import pandas as pd
from pathlib import Path

arousal = pd.read_csv("data/arousal.csv")
valence = pd.read_csv("data/valence.csv")

arousal.columns = arousal.columns.str.strip()
valence.columns = valence.columns.str.strip()

# Get the time-sample columns
arousal_samples = [
    col for col in arousal.columns if col.startswith("sample_")
]
valence_samples = [
    col for col in valence.columns if col.startswith("sample_")
]

print("AROUSAL")
print("Songs:", arousal["song_id"].nunique())
print("Time samples:", len(arousal_samples))
print("First sample:", arousal_samples[0])
print("Last sample:", arousal_samples[-1])

print("\nVALENCE")
print("Songs:", valence["song_id"].nunique())
print("Time samples:", len(valence_samples))
print("First sample:", valence_samples[0])
print("Last sample:", valence_samples[-1])

# Check how many valid labels each song has
arousal["valid_samples"] = arousal[arousal_samples].notna().sum(axis=1)
valence["valid_samples"] = valence[valence_samples].notna().sum(axis=1)

print("\nVALID SAMPLES PER SONG")
print("Arousal:")
print(arousal["valid_samples"].describe().round(1))

print("\nValence:")
print(valence["valid_samples"].describe().round(1))

# Check whether the time columns match
print("\nTIME COLUMNS MATCH:", arousal_samples == valence_samples)

# Show sample coverage for the first 10 songs
print("\nFIRST 10 SONGS")
print(
    arousal[["song_id", "valid_samples"]]
    .head(10)
    .to_string(index=False)
)

# Check whether song IDs match
print(
    "\nSONG IDs MATCH:",
    set(arousal["song_id"]) == set(valence["song_id"])
)

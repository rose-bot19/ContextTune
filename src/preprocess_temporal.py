
from pathlib import Path
import pandas as pd

# Paths
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"

FEATURE_FOLDER = Path(
    r"C:\Users\esorf\Downloads\features\features"
)

# Load dynamic annotations
arousal = pd.read_csv(DATA / "arousal.csv")
valence = pd.read_csv(DATA / "valence.csv")

arousal.columns = arousal.columns.str.strip()
valence.columns = valence.columns.str.strip()

# Match songs using song_id
arousal = arousal.set_index("song_id")
valence = valence.set_index("song_id")

# Use only timestamps available in BOTH annotation files
arousal_times = {
    col: int(col.replace("sample_", "").replace("ms", ""))
    for col in arousal.columns
    if col.startswith("sample_")
}

valence_times = {
    col: int(col.replace("sample_", "").replace("ms", ""))
    for col in valence.columns
    if col.startswith("sample_")
}

common_times = sorted(
    set(arousal_times.values()) & set(valence_times.values())
)

# Map timestamp in milliseconds to annotation column
arousal_by_time = {
    ms: col for col, ms in arousal_times.items()
}
valence_by_time = {
    ms: col for col, ms in valence_times.items()
}

records = []
songs_processed = 0
songs_skipped = 0

# Process each song
for song_id in arousal.index:
    feature_path = FEATURE_FOLDER / f"{song_id}.csv"

    if not feature_path.exists() or song_id not in valence.index:
        songs_skipped += 1
        continue

    # Original feature files use semicolons
    features = pd.read_csv(feature_path, sep=";")
    features.columns = features.columns.str.strip()

    if "frameTime" not in features.columns:
        songs_skipped += 1
        continue

    features["time_ms"] = (
        (features["frameTime"] * 1000).round().astype(int)
    )

    # Match acoustic features and annotations by timestamp
    features = features.set_index("time_ms")

    for time_ms in common_times:
        if time_ms not in features.index:
            continue

        a_col = arousal_by_time[time_ms]
        v_col = valence_by_time[time_ms]

        a_value = arousal.loc[song_id, a_col]
        v_value = valence.loc[song_id, v_col]

        # Do not invent labels for missing annotation values
        if pd.isna(a_value) or pd.isna(v_value):
            continue

        # A timestamp should correspond to one feature row
        feature_row = features.loc[time_ms]

        if isinstance(feature_row, pd.DataFrame):
            feature_row = feature_row.iloc[0]

        record = feature_row.drop(
            labels=["frameTime"], errors="ignore"
        ).to_dict()

        record["song_id"] = song_id
        record["time_seconds"] = time_ms / 1000
        record["valence"] = float(v_value)
        record["arousal"] = float(a_value)

        records.append(record)

    songs_processed += 1

# Create and save the temporal dataset
temporal = pd.DataFrame(records)

if temporal.empty:
    raise ValueError(
        "No aligned samples found. Check file paths and timestamps."
    )

output_path = DATA / "contexttune_temporal_dataset.csv"
temporal.to_csv(output_path, index=False)

print("\nTEMPORAL PREPROCESSING COMPLETE")
print("Songs processed:", songs_processed)
print("Songs skipped:", songs_skipped)
print("Dataset shape:", temporal.shape)
print("Unique songs:", temporal["song_id"].nunique())
print("Missing values:", temporal.isna().sum().sum())
print("Time range:", temporal["time_seconds"].min(),
      "to", temporal["time_seconds"].max(), "seconds")
print("Saved to:", output_path)
print("\nFirst 5 rows:")
print(temporal.iloc[:5, :8].to_string(index=False))



import pandas as pd

# Load the metadata that parsed successfully
metadata = pd.read_csv("data/metadata/metadata_2013.csv")

# Remove tabs and extra spaces from column names and text
metadata.columns = metadata.columns.str.strip()

for column in metadata.select_dtypes(include="object").columns:
    metadata[column] = metadata[column].str.strip()

# Convert IDs into numbers
metadata["song_id"] = pd.to_numeric(
    metadata["song_id"], errors="coerce"
)
metadata = metadata.dropna(subset=["song_id"])
metadata["song_id"] = metadata["song_id"].astype(int)

# Load our existing dataset
dataset = pd.read_csv("data/contexttune_dataset.csv")
dataset["song_id"] = dataset["song_id"].astype(int)

# Compare the IDs
metadata_ids = set(metadata["song_id"])
dataset_ids = set(dataset["song_id"])
matching_ids = metadata_ids.intersection(dataset_ids)

print("Metadata rows:", len(metadata))
print("Unique metadata IDs:", len(metadata_ids))
print("Unique dataset IDs:", len(dataset_ids))
print("Matching IDs:", len(matching_ids))

print("\nExample metadata matches:")
print(
    metadata[metadata["song_id"].isin(matching_ids)][
        ["song_id", "Artist", "Song title", "Genre"]
    ].head(10).to_string(index=False)
)

print("\nDo our current recommendations have metadata?")
for song_id in [1523, 343, 1720, 254, 306]:
    match = metadata[metadata["song_id"] == song_id]

    if len(match) > 0:
        print(
            song_id, "->",
            match.iloc[0]["Artist"],
            "-",
            match.iloc[0]["Song title"]
        )
    else:
        print(song_id, "-> No match in metadata_2013.csv")

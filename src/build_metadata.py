
import pandas as pd

# Load the 2013 metadata
metadata_2013 = pd.read_csv("data/metadata/metadata_2013.csv")

metadata_2013 = metadata_2013.rename(columns={
    "Artist": "artist",
    "Song title": "title",
    "Genre": "genre"
})

metadata_2013 = metadata_2013[
    ["song_id", "artist", "title", "genre"]
]

# Load the 2015 metadata, retaining the rows that parse cleanly
metadata_2015 = pd.read_csv(
    "data/metadata/metadata_2015.csv",
    engine="python",
    on_bad_lines="skip"
)

metadata_2015 = metadata_2015.rename(columns={
    "id": "song_id"
})

metadata_2015 = metadata_2015[
    ["song_id", "artist", "title", "genre"]
]

# Combine the two sources
metadata = pd.concat(
    [metadata_2013, metadata_2015],
    ignore_index=True
)

# Clean IDs and text
metadata["song_id"] = pd.to_numeric(
    metadata["song_id"], errors="coerce"
)

metadata = metadata.dropna(subset=["song_id"])
metadata["song_id"] = metadata["song_id"].astype(int)

for column in ["artist", "title", "genre"]:
    metadata[column] = (
        metadata[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

# Keep only records with a usable title
metadata = metadata[metadata["title"] != ""]

# Save the combined metadata
metadata = metadata.drop_duplicates(
    subset=["song_id"], keep="first"
)

metadata.to_csv("data/song_metadata.csv", index=False)

print("Saved data/song_metadata.csv")
print("Metadata records:", len(metadata))
print("\nExample records:")
print(metadata.head(10).to_string(index=False))

# Check coverage against our dataset
dataset = pd.read_csv("data/contexttune_dataset.csv")
dataset["song_id"] = dataset["song_id"].astype(int)

matched = set(metadata["song_id"]) & set(dataset["song_id"])

print("\nDataset songs:", dataset["song_id"].nunique())
print("Songs with metadata:", len(matched))

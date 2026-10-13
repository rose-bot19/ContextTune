
import pandas as pd
import joblib
import numpy as np

# 1. Load the data, model, and song metadata
data = pd.read_csv("data/contexttune_temporal_dataset.csv")
metadata = pd.read_csv("data/song_metadata.csv")
model = joblib.load("src/contexttune_temporal_model.pkl")

# 2. Define example emotional targets on the dynamic scale
emotions = {
    "sad": (-0.6, -0.3),
    "calm": (0.1, -0.6),
    "neutral": (0.0, 0.0),
    "hopeful": (0.4, 0.1),
    "happy": (0.6, 0.3),
    "excited": (0.5, 0.8),
    "angry": (-0.6, 0.7)
}

# 3. Predict emotions using the trained model
feature_columns = data.drop(
    columns=["song_id", "time_seconds", "valence", "arousal"]
).columns

print("Predicting emotions across songs...")
predictions = model.predict(data[feature_columns])

data["predicted_valence"] = predictions[:, 0]
data["predicted_arousal"] = predictions[:, 1]

# 4. Get the user's desired emotional journey
print("\nAvailable example moods:", ", ".join(emotions))
journey_text = input(
    "\nWhat do you feel like listening to? "
    "(For now, use mood names separated by commas): "
).strip().lower()

journey = [
    mood.strip()
    for mood in journey_text.split(",")
    if mood.strip()
]

if not journey or any(mood not in emotions for mood in journey):
    print("Please use the available mood names, separated by commas.")

else:
    target_path = np.array([emotions[mood] for mood in journey])
    song_scores = []

    # 5. Compare each song's predicted emotional journey
    for song_id, song in data.groupby("song_id", sort=False):
        song = song.sort_values("time_seconds")

        predicted_path = song[
            ["predicted_valence", "predicted_arousal"]
        ].to_numpy()

        segments = np.array_split(predicted_path, len(journey))

        song_path = np.array([
            segment.mean(axis=0)
            for segment in segments
            if len(segment) > 0
        ])

        if len(song_path) != len(target_path):
            continue

        distances = np.sqrt(
            np.sum((song_path - target_path) ** 2, axis=1)
        )

        song_scores.append({
            "song_id": int(song_id),
            "trajectory_distance": distances.mean(),
            "predicted_valence": song["predicted_valence"].mean(),
            "predicted_arousal": song["predicted_arousal"].mean()
        })

    # 6. Rank songs, then attach verified metadata
    recommendations = pd.DataFrame(song_scores)

    recommendations = recommendations.merge(
        metadata,
        on="song_id",
        how="inner"
    )

    recommendations = recommendations.sort_values(
        "trajectory_distance"
    ).drop_duplicates(subset=["song_id"]).head(10)

    # 7. Display actual track information
    if recommendations.empty:
        print("No recommendations with matching metadata were found.")
    else:
        print("\nYour requested journey:", " -> ".join(journey))
        print("\nYour top music recommendations:\n")

        
from urllib.parse import quote_plus

for rank, (_, song) in enumerate(
    recommendations.iterrows(), start=1
        ):
        title = str(song["title"]).strip()
        artist = str(song["artist"]).strip()
        genre = str(song["genre"]).strip() or "Unknown"

        search_query = quote_plus(f"{title} {artist}")
        spotify_link = (
        "https://open.spotify.com/search/" + search_query
        )

        print(f"{rank}. {title} — {artist}")
        print(f"   Genre: {genre}")
        print(f"   Listen / find track: {spotify_link}")
        print(
            f"   Emotional match distance: "
            f"{song['trajectory_distance']:.3f}\n"
        )


        print(
            "Note: These tracks are ranked by predicted emotional fit. "
            "Playback links are not connected yet."
        )

import pandas as pd
import joblib

# 1. Load the temporal dataset
data = pd.read_csv("data/contexttune_temporal_dataset.csv")

# 2. Load the trained temporal model
model = joblib.load("src/contexttune_temporal_model.pkl")

# 3. Select one song
song_id = data["song_id"].iloc[0]
song = data[data["song_id"] == song_id].copy()

# 4. Prepare the acoustic features
X = song.drop(
    columns=["song_id", "time_seconds", "valence", "arousal"]
)

# 5. Predict valence and arousal at each time point
predictions = model.predict(X)

song["predicted_valence"] = predictions[:, 0]
song["predicted_arousal"] = predictions[:, 1]

# 6. Display actual and predicted emotions
print("Song ID:", song_id)
print("Number of time points:", len(song))

print(
    song[
        [
            "time_seconds",
            "valence",
            "predicted_valence",
            "arousal",
            "predicted_arousal",
        ]
    ].head(15).round(3).to_string(index=False)
)
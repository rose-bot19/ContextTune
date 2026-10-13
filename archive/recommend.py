
import pandas as pd
import joblib

# 1. Load the dataset and trained model
data = pd.read_csv("data/contexttune_dataset.csv")
model = joblib.load("src/contexttune_model.pkl")

data.columns = data.columns.str.strip()

# 2. Prepare the same input features used during training
X = data.drop(
    columns=[
        "song_id",
        "valence_mean",
        "valence_std",
        "arousal_mean",
        "arousal_std"
    ]
)

# 3. Predict valence and arousal for every song
predictions = model.predict(X)

data["predicted_valence"] = predictions[:, 0]
data["predicted_arousal"] = predictions[:, 1]

# 4. Define target emotions on the DEAM 1-9 scale
moods = {
    "happy": (7.5, 6.0),
    "excited": (7.5, 8.0),
    "calm": (6.0, 2.5),
    "sad": (2.5, 3.0),
    "angry": (2.5, 8.0),
    "neutral": (5.0, 5.0)
}

print("Available moods:", ", ".join(moods.keys()))
mood = input("What mood are you looking for? ").strip().lower()

if mood not in moods:
    print("Mood not recognised. Please choose one from the list.")
else:
    target_valence, target_arousal = moods[mood]

    # 5. Calculate distance from the desired emotion
    data["mood_distance"] = (
        (data["predicted_valence"] - target_valence) ** 2
        + (data["predicted_arousal"] - target_arousal) ** 2
    ) ** 0.5

    # 6. Rank songs: smaller distance means a closer match
    recommendations = data.sort_values(
        by="mood_distance"
    ).head(10)

    print(f"\nTop 10 songs for: {mood}")
    print(
        recommendations[
            ["song_id", "predicted_valence",
             "predicted_arousal", "mood_distance"]
        ].round(2).to_string(index=False)
    )

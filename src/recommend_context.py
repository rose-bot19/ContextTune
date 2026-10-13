
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from urllib.parse import quote_plus

# --------------------------------------------------
# 1. Project paths and API setup
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your project .env file."
    )

client = genai.Client(api_key=API_KEY)

DATA_PATH = PROJECT_DIR / "data" / "contexttune_temporal_dataset.csv"
METADATA_PATH = PROJECT_DIR / "data" / "song_metadata.csv"
MODEL_PATH = PROJECT_DIR / "src" / "contexttune_temporal_model.pkl"

# --------------------------------------------------
# 2. Ask Gemini to interpret the listener's request
# --------------------------------------------------

def interpret_request(user_request):
    prompt = f"""
You are the emotion interpreter for ContextTune, a music recommender.

Interpret the listener's request and create exactly 3 emotional stages
representing how the music should feel from beginning to end.

Valence represents negative-to-positive emotion.
Arousal represents low-to-high energy.

Both values must be between -1.0 and 1.0.
Use nuanced values rather than only extreme values.
The stages should reflect the listener's requested emotional journey.
If no journey is described, create a coherent journey matching the
overall vibe.

Listener request:
{user_request}

Return a short vibe summary and exactly 3 target points.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "vibe_summary": {"type": "STRING"},
            "target_path": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "valence": {"type": "NUMBER"},
                        "arousal": {"type": "NUMBER"}
                    },
                    "required": ["valence", "arousal"]
                }
            }
        },
        "required": ["vibe_summary", "target_path"]
    }

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=250
        )
    )

    import json
    result = json.loads(response.text)

    path = result["target_path"]

    if len(path) != 3:
        raise ValueError("Gemini did not return exactly 3 stages.")

    for point in path:
        for key in ("valence", "arousal"):
            value = float(point[key])
            if not -1 <= value <= 1:
                raise ValueError(
                    f"Gemini returned an invalid {key} value."
                )
            point[key] = value

    return result


# --------------------------------------------------
# 3. Predict song emotions over time
# --------------------------------------------------

def recommend_songs(target_path, top_n=10):
    import joblib

    data = pd.read_csv(DATA_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    model = joblib.load(MODEL_PATH)
    print("[2/3] Dataset, metadata, and trained model loaded.")

    data.columns = data.columns.str.strip()
    metadata.columns = metadata.columns.str.strip()

    # These columns describe a song/time point or contain its labels.
    excluded = {
        "song_id", "time_seconds", "valence", "arousal"
    }

    feature_columns = [
        column for column in data.columns
        if column not in excluded
    ]

    X = data[feature_columns]

    # The trained model predicts valence and arousal for each time point.
    predictions = model.predict(X)
    print("[2/3] Predicted emotions across song time points.")
    print("[3/3] Comparing songs with the requested journey...")

    data["predicted_valence"] = predictions[:, 0]
    data["predicted_arousal"] = predictions[:, 1]

    # Compare the predicted emotional timeline with the 3 target stages.
    results = []

    for song_id, song in data.groupby("song_id", sort=False):
        song = song.sort_values("time_seconds").reset_index(drop=True)

        if len(song) < 3:
            continue

        # Split each song into beginning, middle, and end.
        segments = np.array_split(np.arange(len(song)), 3)
        distances = []

        for indices, target in zip(segments, target_path):
            segment = song.iloc[indices]

            predicted_v = segment["predicted_valence"].mean()
            predicted_a = segment["predicted_arousal"].mean()

            distance = np.sqrt(
                (predicted_v - target["valence"]) ** 2
                + (predicted_a - target["arousal"]) ** 2
            )

            distances.append(distance)

        results.append({
            "song_id": int(song_id),
            "beginning_distance": float(distances[0]),
            "middle_distance": float(distances[1]),
            "ending_distance": float(distances[2]),
            "trajectory_distance": float(np.mean(distances))
        })

    scores = pd.DataFrame(results)

    # Keep only tracks for which we have usable metadata.
    metadata["song_id"] = pd.to_numeric(
        metadata["song_id"], errors="coerce"
    )
    metadata = metadata.dropna(subset=["song_id"])
    metadata["song_id"] = metadata["song_id"].astype(int)

    recommendations = scores.merge(
        metadata,
        on="song_id",
        how="inner"
    )

    recommendations = recommendations.sort_values(
        "trajectory_distance"
    ).head(top_n)

    return recommendations


def describe_emotion(valence, arousal):
    """Translate numerical emotion predictions into plain language."""

    # Describe the emotional tone.
    if valence <= -0.35:
        tone = "melancholic"
    elif valence < -0.10:
        tone = "slightly somber"
    elif valence < 0.20:
        tone = "neutral or reflective"
    elif valence < 0.50:
        tone = "gently positive"
    elif valence < 0.75:
        tone = "happy and uplifting"
    else:
        tone = "strongly positive and euphoric"

    # Describe the energy level.
    if arousal <= -0.35:
        energy = "very calm"
    elif arousal < -0.10:
        energy = "relaxed"
    elif arousal < 0.25:
        energy = "gently paced"
    elif arousal < 0.50:
        energy = "moderately energetic"
    elif arousal < 0.75:
        energy = "high-energy"
    else:
        energy = "intense and exhilarating"

    return f"{tone.capitalize()}, {energy.lower()}"



# --------------------------------------------------
# 4. Run ContextTune
# --------------------------------------------------

def main():
    print("\n=== ContextTune: Emotion-Aware Music ===\n")

    user_request = input(
        "Describe the music you're in the mood for:\n> "
    ).strip()

    if not user_request:
        print("Please enter a description.")
        return

    try:
        print("\n[1/3] Interpreting your request with Gemini...")
        interpretation = interpret_request(user_request)
        print("[1/3] Emotional targets generated successfully.")
        target_path = interpretation["target_path"]

        print("\nYour vibe:")
        print(interpretation["vibe_summary"])

        print("\nEmotional journey:")
        for index, point in enumerate(target_path, start=1):
            print(
                f"Stage {index}: "
                f"valence={point['valence']:.2f}, "
                f"arousal={point['arousal']:.2f}"
            )

        print("\nFinding matching tracks...")

        recommendations = recommend_songs(target_path)

        if recommendations.empty:
            print("No tracks with matching metadata were found.")
            return

        print("\nYour recommendations:\n")

        
        
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
            youtube_link = (
                "https://www.youtube.com/results?search_query="
                + search_query
            )

            print(f"\n{rank}. {title} — {artist}")
            print(f"   Genre: {genre}")
            print(f"   Spotify: {spotify_link}")
            print(f"   YouTube: {youtube_link}")

            print("\n   Emotional journey match:")

            stages = [
                ("Beginning", "beginning_distance"),
                ("Middle", "middle_distance"),
                ("Ending", "ending_distance"),
            ]

            for label, column in stages:
                distance = float(song[column])

                if distance < 0.20:
                    description = "Very close to your requested vibe"
                elif distance < 0.40:
                    description = "Fairly close to your requested vibe"
                elif distance < 0.60:
                    description = "A partial match to your requested vibe"
                else:
                    description = "A weaker match to your requested vibe"

                print(f"   {label}: {description}.")

            overall = float(song["trajectory_distance"])
            print(
                f"\n   Overall trajectory match score: {overall:.3f}"
            )
            print(
                "   Lower scores indicate a closer match to your "
                "requested emotional journey."
            )


    except Exception as error:
        print(f"\nContextTune encountered an error: {error}")
        print(
            "Check your API quota, model availability, "
            "file paths, and installed packages."
        )


if __name__ == "__main__":
    main()

# ContextTune 

**An emotion-aware music recommender powered by machine learning and natural-language understanding.**

ContextTune explores a different way to discover music: instead of choosing only a genre or mood label, users can describe how they want to feel.

For example:

- "I want to feel like I'm on a rollercoaster."
- "Recommend music for a hopeful road trip."
- "I want something energetic that gradually winds down."

ContextTune uses Gemini to interpret a natural-language request as an emotional journey, then uses a trained machine-learning model to estimate song emotions over time and rank tracks by their similarity to the requested journey.

# Features

- **Natural-language mood input:** Describe a feeling or scenario in your own words.
- **Emotional journey generation:** Gemini generates three target stages: beginning, middle, and ending.
- **Temporal emotion prediction:** A Random Forest regression model estimates valence and arousal across song time points.
- **Trajectory-based recommendations:** Songs are ranked using Euclidean distance between their predicted emotional trajectory and the requested trajectory.
- **Music discovery links:** Open Spotify and YouTube search results for recommended tracks.

# How it works

1. **User input:** The user describes a desired musical experience.
2. **Emotion interpretation:** Gemini converts the description into three emotional targets represented by valence and arousal.
3. **Emotion prediction:** The trained Random Forest model estimates valence and arousal for song time points.
4. **Trajectory comparison:** ContextTune compares each song's beginning, middle, and ending against the requested emotional journey.
5. **Ranking:** Songs with lower trajectory distances are ranked as closer matches.
6. **Music discovery:** The application provides Spotify and YouTube search links.

# Valence and arousal

- **Valence** represents the positive or negative character of an emotion.
- **Arousal** represents its energy or intensity.

The model's numerical outputs are simplified representations of emotion, not definitive descriptions of how every listener will experience a song.

# Tech stack

- Python
- pandas and NumPy
- scikit-learn
- Random Forest regression
- Gemini API through Google's Python SDK
- python-dotenv
- DEAM music emotion annotations

#Project structure

```text
ContextTune/
├── data/
│   ├── metadata/
│   └── song_metadata.csv
├── src/
│   ├── recommend_context.py
│   ├── recommend_temporal.py
│   ├── recommend.py
│   ├── train_temporal.py
│   ├── train_model.py
│   ├── preprocess_temporal.py
│   └── build_metadata.py
├── notebooks/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Additional datasets and trained model files may be generated locally and are not necessarily included in the repository.

# Setup

# 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ContextTune.git
cd ContextTune
```

# 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

# 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

# 4. Configure the API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_actual_api_key_here
```

Obtain a key through Google AI Studio. Never commit your `.env` file or expose your API key publicly.

#5. Prepare the required data and model

The application requires the compatible temporal dataset, song metadata, and trained temporal model at the file paths expected by the source code. These artifacts must be obtained or generated before running the application.

# 6. Run ContextTune

```bash
python src/recommend_context.py
```

# Model evaluation

The temporal Random Forest model was evaluated using a song-grouped train/test split to reduce the risk of the same song appearing in both sets.

| Target | Mean Absolute Error (MAE) | R² |
|---|---:|---:|
| Valence | 0.1715 | 0.2791 |
| Arousal | 0.1529 | 0.4980 |

These results describe performance on the held-out evaluation data. They do not guarantee that the recommended songs will match every user's subjective experience.

# Limitations

- Emotion labels and interpretations are inherently subjective.
- Recommendation quality depends on the available dataset, model predictions, and metadata.
- Some song titles, artists, or genres may be incomplete or inaccurate.
- Spotify and YouTube links are search links, not guarantees of track availability.
- Gemini's interpretation may vary between requests.
- The application currently uses three emotional stages to represent a more complex musical experience.

# Future improvements

- Improve song metadata quality and track identification.
- Evaluate recommendation quality with human feedback.
- Improve the model's ability to capture changes in emotion throughout a song.
- Explore optional Spotify integration.
- Build a user-friendly web interface.

# Project purpose

ContextTune is a machine-learning project exploring the intersection of natural-language understanding, temporal emotion prediction, and personalized music discovery.
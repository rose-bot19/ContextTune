
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv("data/contexttune_dataset.csv")

# 2. Separate input features (X) and target values (y)
data.columns = data.columns.str.strip()

X = data.drop(
    columns=[
        "song_id",
        "valence_mean",
        "valence_std",
        "arousal_mean",
        "arousal_std"
    ]
)

y = data[["valence_mean", "arousal_mean"]]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Create and train the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Predict emotion ratings for unseen songs
predictions = model.predict(X_test)

# 6. Evaluate predictions
for i, target in enumerate(["Valence", "Arousal"]):
    mae = mean_absolute_error(y_test.iloc[:, i], predictions[:, i])
    r2 = r2_score(y_test.iloc[:, i], predictions[:, i])

    print(f"\n{target} results")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

print("\nActual vs predicted examples:")
for i in range(5):
    print(
        f"Actual: {y_test.iloc[i].values.round(2)} | "
        f"Predicted: {predictions[i].round(2)}"
    )


# Save the trained model
joblib.dump(model, "src/contexttune_model.pkl")

print("\nModel saved successfully!")

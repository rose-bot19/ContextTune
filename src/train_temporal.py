
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the preprocessed temporal dataset
data = pd.read_csv("data/contexttune_temporal_dataset.csv")

# Define acoustic inputs and emotion targets
target_columns = ["valence", "arousal"]

X = data.drop(
    columns=["song_id", "time_seconds", "valence", "arousal"]
)

y = data[target_columns]
groups = data["song_id"]

# Split by SONG, so no song appears in both sets
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("Training songs:", groups.iloc[train_indices].nunique())
print("Testing songs:", groups.iloc[test_indices].nunique())
print("Training time points:", len(X_train))
print("Testing time points:", len(X_test))

# Train the temporal prediction model
model = RandomForestRegressor(
    n_estimators=60,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=2
)

print("\nTraining temporal model...")
model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)

for i, target in enumerate(target_columns):
    mae = mean_absolute_error(
        y_test.iloc[:, i],
        predictions[:, i]
    )
    r2 = r2_score(
        y_test.iloc[:, i],
        predictions[:, i]
    )

    print(f"\n{target.capitalize()} results")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

# Save the trained model
joblib.dump(model, "src/contexttune_temporal_model.pkl")

print("\nTemporal model saved successfully!")

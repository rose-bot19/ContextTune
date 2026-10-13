
from pathlib import Path
import pandas as pd

# Original folder containing the individual song feature CSVs
feature_folder = Path(
    r"C:\Users\esorf\Downloads\features\features"
)

files = [
    f for f in feature_folder.glob("*.csv")
    if f.name != "contexttune_dataset.csv"
]

print("Feature CSV files found:", len(files))
print("First 10 filenames:")
for file in files[:10]:
    print(" ", file.name)

# Find the original feature file for song 2
matches = [
    f for f in files
    if f.stem == "2"
]

if not matches:
    print("\nNo file named 2.csv found.")
    print("Check the filenames printed above.")
else:
    file = matches[0]
    features = pd.read_csv(file, sep=";")
    features.columns = features.columns.str.strip()

    print("\nSONG 2 FEATURE FILE")
    print("Filename:", file.name)
    print("Shape:", features.shape)
    print("Columns:", features.columns[:8].tolist())

    if "frameTime" in features.columns:
        times = features["frameTime"].dropna()

        print("\nFrame times:")
        print("First 10:", times.head(10).tolist())
        print("Minimum:", times.min())
        print("Maximum:", times.max())

        print("\nTime differences:")
        print(times.diff().dropna().describe())

    print("\nFirst 5 rows:")
    print(features.head().to_string(index=False))


import pandas as pd

files = [
    "data/metadata/metadata_2014.csv",
    "data/metadata/metadata_2015.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)

    try:
        df = pd.read_csv(
            file,
            sep=",",
            engine="python",
            on_bad_lines="warn"
        )

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head(5).to_string(index=False))

    except Exception as e:
        print("Error:", e)

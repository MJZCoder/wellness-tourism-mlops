
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 3. Remove unnecessary columns
# --------------------------------------------------

df = df.drop(
    columns=["CustomerID", "Unnamed: 0"],
    errors="ignore"
)

# Remove rows without a valid binary target
df = df[df["ProdTaken"].isin([0, 1])].copy()

print("Dataset shape after cleaning:", df.shape)

# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# --------------------------------------------------
# 5. Split dataset (80% training / 20% testing)
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 6. Recombine features and target for CSV storage
# --------------------------------------------------

train_df = X_train.copy()
train_df["ProdTaken"] = y_train

test_df = X_test.copy()
test_df["ProdTaken"] = y_test

# --------------------------------------------------
# 7. Save split datasets
# --------------------------------------------------

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nData preparation completed successfully!")

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

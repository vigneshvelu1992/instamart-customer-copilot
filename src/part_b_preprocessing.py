
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "lapse_cohort.csv"
TRAIN_FILE = ROOT / "outputs" / "lapse_train.csv"
TEST_FILE = ROOT / "outputs" / "lapse_test.csv"

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category"
]


def main():
    print("=" * 60)
    print("PART B - TASK 5: FEATURE PREPROCESSING")
    print("=" * 60)

    # 1. Load the cohort
    df = pd.read_csv(INPUT_FILE)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["lapsed"]

    # 2. Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\nTRAIN / TEST SPLIT")
    print("Total customers:", len(df))
    print("Training customers:", len(X_train))
    print("Testing customers:", len(X_test))

    print("\nTRAINING CLASS DISTRIBUTION")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTEST CLASS DISTRIBUTION")
    print(y_test.value_counts().sort_index().to_string())

    # 3. Create preprocessing transformations
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # 4. Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data without fitting again
    X_test_processed = preprocessor.transform(X_test)

    # 5. Verify frequency scaling
    frequency_index = NUMERIC_FEATURES.index("frequency")

    full_frequency_mean = X["frequency"].mean()

    train_frequency_mean = (
        preprocessor
        .named_transformers_["numeric"]
        .mean_[frequency_index]
    )

    print("\nFREQUENCY MEAN COMPARISON")
    print("Full cohort frequency mean:", round(full_frequency_mean, 4))
    print("Training scaler frequency mean:", round(train_frequency_mean, 4))

    # 6. Verify feature dimensions
    print("\nPROCESSED DATA DIMENSIONS")
    print("Training rows:", X_train_processed.shape[0])
    print("Training features:", X_train_processed.shape[1])
    print("Testing rows:", X_test_processed.shape[0])
    print("Testing features:", X_test_processed.shape[1])

    # 7. Check missing values and feature columns
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    assert list(X_train.columns) == (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    assert X_train_processed.shape[1] == X_test_processed.shape[1]

    # 8. Save original train/test splits for Task 6
    train_output = X_train.copy()
    train_output["lapsed"] = y_train

    test_output = X_test.copy()
    test_output["lapsed"] = y_test

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    train_output.to_csv(TRAIN_FILE, index=False)
    test_output.to_csv(TEST_FILE, index=False)

    print("\nVALIDATION")
    print("No missing feature values: PASSED")
    print("Train/test feature columns match: PASSED")
    print("Scaler fitted only on training data: PASSED")
    print("Encoder fitted only on training data: PASSED")
    print("Test data transformed without fitting: PASSED")

    print("\n" + "=" * 60)
    print("TASK 5 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Saved training data:", TRAIN_FILE)
    print("Saved testing data:", TEST_FILE)


if __name__ == "__main__":
    main()

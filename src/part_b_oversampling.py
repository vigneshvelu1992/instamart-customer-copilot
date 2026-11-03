
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent

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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    print("=" * 60)
    print("PART B - TASK 7: OVERSAMPLING")
    print("=" * 60)

    # 1. Load original training and testing data
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train[FEATURES]
    y_train = train["lapsed"]

    X_test = test[FEATURES]
    y_test = test["lapsed"]

    # 2. Display original class distribution
    print("\nORIGINAL TRAINING CLASS DISTRIBUTION")
    print(y_train.value_counts().sort_index().to_string())

    # 3. Separate the two training classes
    majority = train[train["lapsed"] == 0]
    minority = train[train["lapsed"] == 1]

    # 4. Duplicate lapsed training customers with replacement
    minority_oversampled = minority.sample(
        n=len(majority) - len(minority),
        replace=True,
        random_state=42
    )

    # 5. Combine original training rows and duplicated rows
    balanced_train = pd.concat(
        [train, minority_oversampled],
        ignore_index=True
    )

    # Shuffle the balanced training dataset
    balanced_train = balanced_train.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    X_balanced = balanced_train[FEATURES]
    y_balanced = balanced_train["lapsed"]

    print("\nBALANCED TRAINING CLASS DISTRIBUTION")
    print(y_balanced.value_counts().sort_index().to_string())

    # 6. Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    # 7. Refit the selected Logistic Regression model
    # Task 6 selected C=0.1
    balanced_model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=0.1,
                max_iter=1000
            )
        )
    ])

    balanced_model.fit(X_balanced, y_balanced)

    # 8. Evaluate on the original, unchanged test data
    predictions = balanced_model.predict(X_test)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\nOVERSAMPLED MODEL TEST RESULTS")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

    # 9. Compare with original Task 6 Logistic Regression
    print("\nTASK 6 VS TASK 7 COMPARISON")
    print("Original Logistic Regression:")
    print("Precision: 0.6545")
    print("Recall:    0.8090")
    print("F1 score:  0.7236")

    print("\nOversampled Logistic Regression:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

    print("\nVALIDATION")
    assert len(balanced_train) == 1140
    assert y_balanced.value_counts()[0] == 570
    assert y_balanced.value_counts()[1] == 570
    assert len(X_test) == 232

    print("Balanced training rows: PASSED")
    print("Equal class counts: PASSED")
    print("Original test set unchanged: PASSED")

    print("\n" + "=" * 60)
    print("TASK 7 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
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


def create_preprocessor():
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name} TEST RESULTS")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    print("=" * 60)
    print("PART B - TASK 6: LAPSE PREDICTION MODELS")
    print("=" * 60)

    # 1. Load the saved training and testing datasets
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train[FEATURES]
    y_train = train["lapsed"]

    X_test = test[FEATURES]
    y_test = test["lapsed"]

    # 2. Baseline: predict every customer as not lapsed
    baseline_predictions = [0] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    baseline_f1 = f1_score(
        y_test, baseline_predictions, zero_division=0
    )

    print("\nBASELINE RESULTS")
    print("Baseline accuracy:", round(baseline_accuracy, 4))
    print("Baseline lapsed-class F1:", round(baseline_f1, 4))

    # 3. Logistic Regression Pipeline
    logistic_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    logistic_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100]
    }

    print("\nLOGISTIC REGRESSION GRID SEARCH")
    print("Running 5-fold cross-validation...")

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    logistic_search.fit(X_train, y_train)

    print("Best parameters:", logistic_search.best_params_)
    print("Best CV F1:", round(logistic_search.best_score_, 4))

    # 4. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    rf_grid = {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    }

    print("\nRANDOM FOREST GRID SEARCH")
    print("Running 5-fold cross-validation...")

    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rf_search.fit(X_train, y_train)

    print("Best parameters:", rf_search.best_params_)
    print("Best CV F1:", round(rf_search.best_score_, 4))

    # 5. Evaluate both tuned models on the original test set
    logistic_results = evaluate_model(
        "LOGISTIC REGRESSION",
        logistic_search.best_estimator_,
        X_test,
        y_test
    )

    rf_results = evaluate_model(
        "RANDOM FOREST",
        rf_search.best_estimator_,
        X_test,
        y_test
    )

    # 6. Compare model results
    comparison = pd.DataFrame([
        logistic_results,
        rf_results
    ])

    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    # 7. Select the model with the higher test F1
    if rf_results["f1"] >= logistic_results["f1"]:
        selected_name = "Random Forest"
        selected_model = rf_search.best_estimator_
    else:
        selected_name = "Logistic Regression"
        selected_model = logistic_search.best_estimator_

    selected_f1 = max(
        rf_results["f1"],
        logistic_results["f1"]
    )

    print("\nSELECTED MODEL:", selected_name)
    print("Selected test F1:", round(selected_f1, 4))

    assert selected_f1 >= 0.55, (
        "Selected model test F1 is below the required 0.55."
    )

    print("\nTASK 6 VALIDATION")
    print("Both models trained successfully: PASSED")
    print("Five-fold GridSearchCV completed: PASSED")
    print("Both models evaluated on the test set: PASSED")
    print("Selected model F1 >= 0.55: PASSED")

    print("\n" + "=" * 60)
    print("TASK 6 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

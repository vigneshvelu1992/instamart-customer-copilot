
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================
# PART B - TASK 8: MODEL INSIGHTS AND RETENTION LIST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

COHORT_FILE = OUTPUT_DIR / "lapse_cohort.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

MODEL_FILE = OUTPUT_DIR / "lapse_model.joblib"
RETENTION_FILE = OUTPUT_DIR / "retention_list.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("PART B - TASK 8: MODEL INSIGHTS AND RETENTION LIST")
print("=" * 60)

cohort = pd.read_csv(COHORT_FILE)
segments = pd.read_csv(SEGMENTS_FILE)

print("\nData loaded successfully.")
print("Lapse cohort rows:", len(cohort))
print("Customer segments rows:", len(segments))


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

all_features = numeric_features + categorical_features

required_columns = (
    ["customer_id", "lapsed"]
    + all_features
)

missing_columns = [
    col for col in required_columns
    if col not in cohort.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in lapse_cohort.csv: {missing_columns}"
    )


# ============================================================
# 3. RECREATE THE SAME TRAIN / TEST SPLIT
# ============================================================

X = cohort[all_features].copy()
y = cohort["lapsed"].astype(int)
customer_ids = cohort["customer_id"].copy()

(
    X_train,
    X_test,
    y_train,
    y_test,
    train_customer_ids,
    test_customer_ids
) = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAIN / TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 4. CREATE PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor():

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# ============================================================
# 5. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42
            )
        )
    ]
)

rf_pipeline.fit(X_train, y_train)

feature_names = (
    rf_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importances = (
    rf_pipeline
    .named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(importance_df.head(5).to_string(index=False))


# ============================================================
# 6. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

selected_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "model",
            LogisticRegression(
                C=0.1,
                max_iter=1000
            )
        )
    ]
)

selected_model.fit(X_train, y_train)

logistic_feature_names = (
    selected_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    selected_model
    .named_steps["model"]
    .coef_[0]
)

coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": coefficients
})

print("\nLOGISTIC REGRESSION COEFFICIENTS")

for feature in [
    "numeric__recency",
    "numeric__frequency",
    "numeric__monetary"
]:

    row = coefficient_df[
        coefficient_df["feature"] == feature
    ]

    if not row.empty:

        value = row.iloc[0]["coefficient"]

        direction = (
            "Positive" if value > 0 else "Negative"
        )

        print(
            f"{feature}: {value:.4f} ({direction})"
        )


# ============================================================
# 7. SAVE AND RELOAD SELECTED MODEL
# ============================================================

print("\nMODEL SAVING")

joblib.dump(selected_model, MODEL_FILE)

print("Model saved:", MODEL_FILE)

loaded_model = joblib.load(MODEL_FILE)

print("\nMODEL RELOAD VALIDATION")

original_predictions = selected_model.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

predictions_identical = np.array_equal(
    original_predictions,
    reloaded_predictions
)

print("Predictions identical:", predictions_identical)

assert predictions_identical, (
    "Model predictions changed after reloading!"
)


# ============================================================
# 8. GENERATE LAPSE PROBABILITIES
# ============================================================

probabilities = loaded_model.predict_proba(X_test)[:, 1]


# ============================================================
# 9. RECOVER CUSTOMER IDs AND CREATE TEST PREDICTIONS
# ============================================================

# The customer IDs were kept separately during the same
# train_test_split, so they match the test predictions exactly.

retention = pd.DataFrame({
    "customer_id": test_customer_ids.to_numpy(),
    "lapse_probability": probabilities,
    "actually_lapsed": y_test.to_numpy()
})


# ============================================================
# 10. FILTER PREMIUM CUSTOMERS
# ============================================================

# Convert Premium flag into a consistent boolean value.

segments["is_premium"] = (
    segments["is_premium"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

premium_customers = segments.loc[
    segments["is_premium"],
    ["customer_id"]
].drop_duplicates()

premium_retention = retention.merge(
    premium_customers,
    on="customer_id",
    how="inner"
)


# ============================================================
# 11. RANK TOP 10 PREMIUM CUSTOMERS
# ============================================================

retention_list = (
    premium_retention
    .sort_values(
        by="lapse_probability",
        ascending=False
    )
    .head(10)
    .copy()
)

retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
]

retention_list.to_csv(
    RETENTION_FILE,
    index=False
)


# ============================================================
# 12. VALIDATION
# ============================================================

print("\nRETENTION LIST")

print(
    retention_list.to_string(index=False)
)

print("\nRETENTION LIST SUMMARY")

print(
    "Premium customers in test set:",
    len(premium_retention)
)

print(
    "Customers in retention list:",
    len(retention_list)
)

actual_lapsed_count = int(
    retention_list["actually_lapsed"].sum()
)

print(
    "Actually lapsed customers in top 10:",
    actual_lapsed_count
)

print(
    "Retention list saved:",
    RETENTION_FILE
)

assert len(retention_list) > 0, (
    "No Premium customers found in the test set!"
)

assert len(retention_list) <= 10

assert retention_list["customer_id"].is_unique

assert retention_list["lapse_probability"].is_monotonic_decreasing

assert predictions_identical

print("\nVALIDATION RESULTS")
print("Customer IDs recovered correctly: PASS")
print("Model reload predictions identical: PASS")
print("Premium customer filtering: PASS")
print("Top 10 ranking: PASS")
print("Retention list saved successfully: PASS")

print("\n" + "=" * 60)
print("PART B - TASK 8 COMPLETED SUCCESSFULLY")
print("=" * 60)

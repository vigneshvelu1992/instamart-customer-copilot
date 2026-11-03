
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PART B - TASK 2: K-MEANS CLUSTERING
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "rfm_table.csv"
OUTPUT_FILE = ROOT / "outputs" / "kmeans_evaluation.csv"

FEATURES = ["recency", "frequency", "monetary"]

REFERENCE = {
    2: (1820.8, 0.465),
    3: (1268.2, 0.445),
    4: (743.1, 0.526),
    5: (617.4, 0.481),
    6: (503.2, 0.482),
    7: (408.5, 0.416),
    8: (366.5, 0.412),
}


def main():

    print("=" * 60)
    print("PART B - TASK 2: K-MEANS CLUSTERING")
    print("=" * 60)

    # 1. Load RFM table
    rfm = pd.read_csv(INPUT_FILE)

    X = rfm[FEATURES]

    # 2. Standardise the RFM features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Evaluate K from 2 to 8
    results = []

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        results.append({
            "K": k,
            "inertia": inertia,
            "silhouette": silhouette
        })

    evaluation = pd.DataFrame(results)

    # 4. Display evaluation table
    print("\nK-MEANS EVALUATION RESULTS")

    print(
        evaluation.to_string(
            index=False,
            formatters={
                "inertia": "{:.1f}".format,
                "silhouette": "{:.3f}".format
            }
        )
    )

    # 5. Check reference values
    print("\nREFERENCE CHECKS")

    all_passed = True

    for _, row in evaluation.iterrows():

        k = int(row["K"])
        expected_inertia, expected_silhouette = REFERENCE[k]

        inertia_passed = (
            abs(row["inertia"] - expected_inertia)
            <= expected_inertia * 0.05
        )

        silhouette_passed = (
            abs(row["silhouette"] - expected_silhouette)
            <= 0.05
        )

        passed = inertia_passed and silhouette_passed

        print(
            f"K={k}: {'PASSED' if passed else 'FAILED'} "
            f"(inertia={inertia_passed}, "
            f"silhouette={silhouette_passed})"
        )

        if not passed:
            all_passed = False

    if not all_passed:
        raise AssertionError(
            "One or more K-means reference checks failed."
        )

    # 6. Save evaluation results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # 7. Select final K
    selected_k = 4

    print("\nSELECTED CLUSTER COUNT")
    print(f"Selected K: {selected_k}")

    print("\nBUSINESS INTERPRETATION")
    print(
        "K=4 provides a useful balance between cluster separation "
        "and business interpretability."
    )

    print(
        "K=8 always has lower inertia than K=2 because increasing "
        "the number of clusters allows customers to be grouped "
        "more tightly. However, lower inertia alone does not "
        "identify the best K because it naturally decreases as "
        "more clusters are added and may create unnecessarily "
        "complex segments."
    )

    print("\n" + "=" * 60)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nSaved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

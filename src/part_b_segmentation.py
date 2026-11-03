from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "rfm_table.csv"
OUTPUT_FILE = ROOT / "outputs" / "customer_segments.csv"

FEATURES = ["recency", "frequency", "monetary"]


def main():
    print("=" * 60)
    print("PART B - TASK 3: CUSTOMER SEGMENT PROFILING")
    print("=" * 60)

    # 1. Load the RFM data
    rfm = pd.read_csv(INPUT_FILE)

    # 2. Scale the RFM features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[FEATURES])

    # 3. Fit K-Means using the selected K=4
    kmeans = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42
    )

    rfm["segment_id"] = kmeans.fit_predict(X_scaled)

    # 4. Calculate the average RFM values for each segment
    profile = (
        rfm.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean")
        )
        .reset_index()
    )

    print("\nCUSTOMER SEGMENT PROFILE")
    print(profile.round(2).to_string(index=False))

    # 5. Assign meaningful business names
    # Premium = segment with the highest average monetary value
    premium_id = profile.loc[
        profile["mean_monetary"].idxmax(), "segment_id"
    ]

    remaining = profile[
        profile["segment_id"] != premium_id
    ].copy()

    loyal_id = remaining.loc[
        remaining["mean_frequency"].idxmax(), "segment_id"
    ]

    remaining = remaining[
        remaining["segment_id"] != loyal_id
    ].copy()

    at_risk_id = remaining.loc[
        remaining["mean_recency"].idxmax(), "segment_id"
    ]

    remaining = remaining[
        remaining["segment_id"] != at_risk_id
    ]

    regular_id = remaining.iloc[0]["segment_id"]

    segment_names = {
        int(premium_id): "Premium",
        int(loyal_id): "Loyal",
        int(at_risk_id): "At-Risk",
        int(regular_id): "Regular"
    }

    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = rfm["segment_id"] == premium_id

    print("\nSEGMENT BUSINESS NAMES")
    for segment_id, name in segment_names.items():
        print(f"Segment {segment_id}: {name}")

    # 6. Verify that Premium has the highest average monetary value
    premium_mean = profile.loc[
        profile["segment_id"] == premium_id,
        "mean_monetary"
    ].iloc[0]

    assert premium_mean == profile["mean_monetary"].max()
    assert rfm["is_premium"].sum() == profile.loc[
        profile["segment_id"] == premium_id,
        "customer_count"
    ].iloc[0]

    # 7. Apply PCA to the scaled RFM data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    variance = pca.explained_variance_ratio_

    print("\nPCA EXPLAINED VARIANCE")
    print(f"Principal Component 1: {variance[0]:.3f}")
    print(f"Principal Component 2: {variance[1]:.3f}")
    print(f"Total explained variance: {variance.sum():.3f}")

    # 8. Save the required customer segmentation file
    output = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium"
        ]
    ]

    output.to_csv(OUTPUT_FILE, index=False)

    # 9. Final checks
    assert len(output) == 1175
    assert output["segment_id"].nunique() == 4
    assert output["is_premium"].sum() > 0
    assert output["segment_name"].notna().all()

    print("\nVALIDATION")
    print("Customer count:", len(output))
    print("Number of segments:", output["segment_id"].nunique())
    print("Premium customers:", int(output["is_premium"].sum()))
    print("All validation checks PASSED")

    print("\n" + "=" * 60)
    print("TASK 3 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Saved output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
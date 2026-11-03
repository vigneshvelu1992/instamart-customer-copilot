# Instamart Customer Intelligence & Support Copilot

## Overview

This project develops an end-to-end customer intelligence and support copilot for a fictional Instamart business using synthetic customer, product, order, support-ticket and policy data.

The project is divided into four parts:

- **Part A – Data Foundation:** Data auditing, cleaning, SQLite database creation, SQL queries and exploratory data analysis.
- **Part B – Customer Segments and Lapse Prediction:** RFM analysis, K-means clustering, customer segmentation and machine-learning-based lapse prediction.
- **Part C – Support Ticket Triage with Guardrails:** Privacy-safe, resilient and cost-aware support-ticket classification and routing.
- **Part D – Policy RAG Assistant and Refund Agent:** Retrieval-grounded policy assistance and a tool-based refund agent.

All model interactions use the supplied offline mock language model. No paid API or external model service is required.

## Reproduce

The following commands reproduce the currently completed Parts A and B. Part C and Part D execution commands will be added after their implementations are completed.

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the environment.

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Generate the synthetic datasets:

   ```bash
   python generate_data.py
   ```

5. Audit the raw datasets:

   ```bash
   python src/part_a_audit.py
   ```

6. Clean the datasets:

   ```bash
   python src/part_a_clean.py
   ```

7. Create and populate the SQLite database:

   ```bash
   python src/part_a_database.py
   ```

8. Execute SQL analysis:

   ```bash
   python src/part_a_sql.py
   ```

9. Perform exploratory data analysis:

   ```bash
   python src/part_a_eda.py
   ```

10. Generate the RFM table:

    ```bash
    python src/part_b_rfm.py
    ```

11. Evaluate K-means clustering:

    ```bash
    python src/part_b_clustering.py
    ```

12. Generate customer segment profiles:

    ```bash
    python src/part_b_segmentation.py
    ```

13. Generate the lapse cohort and features:

    ```bash
    python src/part_b_lapse_features.py
    ```

14. Perform preprocessing and train/test splitting:

    ```bash
    python src/part_b_preprocessing.py
    ```

15. Train and evaluate lapse prediction models:

    ```bash
    python src/part_b_lapse_model.py
    ```

16. Evaluate oversampling:

    ```bash
    python src/part_b_oversampling.py
    ```

17. Generate model insights and the retention list:

    ```bash
    python src/part_b_model_insights.py
    ```

## Part A – Data Foundation: Clean, Store, Query

### Objective

Audit and clean the raw customer, product and order datasets, store the cleaned data in SQLite and perform business analysis using SQL and pandas.

### Data audit findings

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending orders | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank gender values | 30 |

### Cleaning results

- Standardized city and gender values.
- Removed exact duplicate order records.
- Excluded pending orders from the cleaned order dataset.
- Corrected invalid and non-numeric values using documented cleaning rules.
- Retained cancelled orders while handling their delivery-time values separately.
- Validated customer ages and missing values.

### Clean dataset summary

| Dataset | Rows |
|---|---:|
| Clean customers | 1,200 |
| Clean products | 40 |
| Clean orders | 12,309 |

Clean order status:

| Status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |

Completed order revenue: **INR 2,998,668.80**

### Database and SQL

Created `data/instamart.db` with the following tables:

- customers
- products
- orders

SQL business analysis is maintained in `sql/queries.sql`.

### Exploratory data analysis

- Completed order amount mean: INR 269.91.
- Median: INR 135.00.
- Skewness: approximately 4.61.
- Upper outliers: 897.
- Largest absolute correlation: order amount and discount percentage, approximately -0.095.

The weak correlation does not establish a causal relationship.

## Part B – Customer Segments and Lapse Prediction

### Objective

Develop actionable customer segments using RFM analysis and predict customer lapse using machine-learning models.

### 1. RFM analysis

Snapshot date: `2025-10-01 00:00:00`

| Metric | Result |
|---|---:|
| Customers analysed | 1,175 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 |
| Maximum frequency | 33 |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

### 2. K-means clustering

StandardScaler was applied to the RFM features. K-means was evaluated for K values from 2 to 8.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Selected cluster count: **K = 4**

Inertia decreases as the number of clusters increases because additional clusters reduce within-cluster distances. However, choosing K based only on inertia can create unnecessary complexity. K=4 provided a strong silhouette score and meaningful business segments.

### 3. Customer segment profiles

| Segment name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) |
|---|---:|---:|---:|---:|
| Regular | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Loyal | 419 | 8.77 | 19.40 | 3,410.02 |
| At-Risk | 182 | 109.04 | 2.20 | 685.99 |

Recommended actions:

- **Regular:** Encourage repeat purchases through personalized offers.
- **Premium:** Provide loyalty benefits and personalized retention campaigns.
- **Loyal:** Reward frequent purchases and maintain engagement.
- **At-Risk:** Launch reactivation campaigns and targeted incentives.

The segment with the highest mean monetary value is marked as Premium. Cluster IDs are labels and do not represent an inherent ranking.

### 4. PCA

Two principal components explained approximately:

| Component | Explained variance |
|---|---:|
| PC1 | 0.634 |
| PC2 | 0.218 |
| Total | 0.852 |

### 5. Lapse cohort

- Cutoff date: `2025-09-01`.
- Cohort customers: 1,156.
- Lapsed customers: 443.
- Non-lapsed customers: 713.
- Lapse rate: 38.32%.

Features were calculated using information available before the cutoff. Segment labels were not used as model features.

### 6. Model training and evaluation

A stratified 80:20 train/test split was used with `random_state=42`.

| Dataset | Rows |
|---|---:|
| Training | 924 |
| Testing | 232 |

Models evaluated:

- Baseline classifier
- Logistic Regression
- Random Forest

Selected model: **Logistic Regression**

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7629 | 0.7672 |
| Precision | 0.6545 | 0.6804 |
| Recall | 0.8090 | 0.7416 |
| F1-score | 0.7236 | 0.7097 |

Logistic Regression was selected because it achieved the higher test F1-score.

### 7. Oversampling

Random oversampling was applied only to the training data.

| Metric | Original Logistic Regression | Oversampled Logistic Regression |
|---|---:|---:|
| Precision | 0.6545 | 0.5870 |
| Recall | 0.8090 | 0.9101 |
| F1-score | 0.7236 | 0.7137 |

Oversampling improved recall but reduced precision and slightly reduced F1-score.

### 8. Model insights and retention list

Top Random Forest features:

| Feature | Importance |
|---|---:|
| Frequency | 0.352171 |
| Monetary | 0.288692 |
| Recency | 0.114974 |
| Cancellation rate | 0.053057 |
| Average discount percentage | 0.040657 |

Selected Logistic Regression coefficient signs:

| Feature | Coefficient | Direction |
|---|---:|---|
| Recency | -0.0285 | Negative |
| Frequency | -1.2811 | Negative |
| Monetary | 0.0295 | Positive |

The selected model was saved and reloaded using joblib. Predictions before and after reloading were identical.

The retention list contains the 10 Premium customers in the test set with the highest predicted lapse probabilities. Two of these 10 customers actually lapsed.

### Part B output files

- `outputs/rfm_table.csv`
- `outputs/kmeans_evaluation.csv`
- `outputs/customer_segments.csv`
- `outputs/lapse_cohort.csv`
- `outputs/lapse_train.csv`
- `outputs/lapse_test.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C – Support Ticket Triage with Guardrails

**Status: Pending implementation.**

This part will implement a support-ticket triage pipeline with:

- Input guardrails for empty and excessively long tickets.
- Redaction of email addresses, phone numbers and order IDs.
- Zero-shot and few-shot classification prompts.
- Retry handling with exponential backoff and jitter.
- JSON and keyword-based output validation.
- Token-budget enforcement.
- Routing based on ticket category and Premium customer status.

Required outputs:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

Results and validation statistics will be added after implementation.

## Part D – Policy RAG Assistant and Refund Agent

**Status: Pending implementation.**

This part will implement:

- Policy document chunking and TF-IDF retrieval.
- Retrieval-grounded question answering.
- Fallback handling for unsupported questions.
- A ReAct-style refund agent.
- Order lookup and policy search tools.
- Refund calculation and escalation.
- Checkpointing and crash recovery.

Required outputs:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` checkpoint directory.

Scenario results, fallback demonstrations and crash-resume results will be documented after implementation.

## Limitations and Next Steps

### Current limitations

1. The Premium flag is based on whole-window RFM segmentation, which includes activity from September 2025. Consequently, the Premium retention list is illustrative and may contain post-cutoff information. The lapse model itself uses only pre-cutoff features.
2. The negative Logistic Regression recency coefficient is counterintuitive and should be interpreted cautiously. Model coefficients describe associations, not causal effects.
3. The lapse model is trained on synthetic data and should be validated on real-world data before production use.
4. The retention list contains predicted lapse probabilities, not guaranteed future customer behaviour.

### Next steps

- Complete Part C support-ticket triage and privacy guardrails.
- Complete Part D policy retrieval and refund-agent workflows.
- Add all remaining execution commands and validation results to the Reproduce section.
- Run the complete project from a clean clone to verify reproducibility.
- Review the final repository contents and all required deliverables before submission.


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


## Part C — Support Ticket Triage with Guardrails

### Objective

Built an offline support-ticket triage pipeline using the supplied `mock_llm.py`. The pipeline classifies customer tickets, protects personally identifiable information (PII), validates model responses, retries transient errors, enforces a token budget, and routes tickets to the appropriate support team.

### Input guardrails and PII protection

Processed 164 tickets in their original input order.

- **Empty tickets:** 4 skipped without a model call.
- **Long tickets:** 4 flagged for review because their estimated input length exceeded the 400-token limit.
- **Eligible tickets:** 156.
- **PII redaction:** 55 email addresses, 91 phone numbers, and 132 order IDs replaced with placeholders — 278 matches in total.
- **PII verification:** Zero matching email, phone, or order-ID patterns remained in the redacted ticket text.

The user message sent to the model contains only the redacted ticket text. The supplied `mock_llm.py` was not modified, and no external LLM API was used.

### Prompting and response validation

Compared zero-shot and few-shot classification. Both prompts require JSON-only output with one of four categories: `billing`, `delivery`, `technical`, or `fraud`. The few-shot prompt includes labeled examples, including a fraud example.

Every response is validated. Valid JSON categories receive `classified_json`; responses containing exactly one identifiable allowed category word receive `classified_keyword`; ambiguous or unsupported outputs receive `flagged_invalid_output` and are sent for human review.

| Final few-shot status | Tickets |
|---|---:|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |
| **Total** | **164** |

### Zero-shot versus few-shot results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets classified | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy on classified tickets | 85.7% | 100.0% |
| Total response tokens | 16,108 | 29,056 |
| Tokens per classified ticket | 109.58 | 197.66 |

The zero-shot confusion table showed 11 fraud tickets incorrectly classified as billing and 10 technical tickets incorrectly classified as billing. Few-shot prompting correctly classified all 147 tickets that received a usable category in this benchmark. It consumed approximately 80.4% more response tokens than zero-shot, but improved measured accuracy by 14.3 percentage points. These results describe the supplied offline mock benchmark, not a guarantee of equivalent accuracy on real-world tickets.

### Retry handling

Each model call is protected by retry handling for transient errors, with up to three retries and exponential backoff with jitter. The mock raised 19 transient errors in each full-budget run; all were recovered. The budget demonstration stopped before exhausting the retry allowance for any ticket.

Retry events are saved to `outputs/triage_retry.log`.

### Token-budget demonstration

The zero-shot budget demonstration used a 6,000-token threshold.

- Tickets classified: 56.
- Tickets marked `budget_exhausted`: 97.
- Output: `outputs/triage_budget_demo.csv` (164 rows).

The final successful response can take the running token total above the threshold when the call began below it. Subsequent calls are blocked once the running total reaches or exceeds the budget.

### Final routing

The final deliverable is `outputs/triage_results.csv`, containing 164 rows and these columns:

`ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`

| Route | Tickets |
|---|---:|
| `support_queue` | 102 |
| `security_team` | 34 |
| `human_review` | 17 |
| `premium_desk` | 11 |
| **Total** | **164** |

Routing uses explicit, rule-based conditions:

- Empty, long, invalid-output, budget-exhausted, and permanent-error statuses are routed to `human_review` with `NORMAL` priority.
- Fraud tickets are routed to `security_team` with `HIGH` priority.
- Other classified tickets whose customer is marked premium in the Part B segmentation output are routed to `premium_desk` with `HIGH` priority.
- Remaining classified tickets are routed to `support_queue` with `NORMAL` priority.
- Customers without a matching segmentation record are treated as non-premium.

### Rule validation versus human review

The pipeline automatically validates input length, redaction, response format, allowed categories, token-budget state, and routing conditions. These checks enforce the defined rules; they do not independently prove that every classification is correct. Tickets with empty or excessive-length input, ambiguous or invalid model responses, exhausted budget, or unrecovered model errors require human review. Fraud classifications are routed to the security team for appropriate handling.

### Part C deliverables

- `src/part_c_triage.py`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_retry.log`

## Part D: Policy RAG and Refund Agent

### 1. Policy document retrieval (RAG)

Created a policy question-answering workflow using five policy documents:

- Delivery policy
- Escalation policy
- Payment policy
- Privacy policy
- Refund policy

Implementation:
- Split policy documents into word-based chunks.
- Tested chunk size of 40 words with 10-word overlap.
- Compared overlapping chunks against non-overlapping chunks.
- Used TF-IDF retrieval to find relevant policy passages.
- Used the supplied local `mock_llm.py` for policy answers.
- Evaluated both approaches against 14 policy questions.
- Out-of-scope questions return a response indicating that the policy documents do not contain the requested information.

Chunking experiment:

| Experiment | Chunks | Evaluation questions | Response tokens |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 15 | 14 | 3,254 |
| 40 words, no overlap | 13 | 14 | 3,190 |

The overlapping approach produced two additional chunks and used 64 more response tokens in this experiment. Some retrieved answers contain text-boundary and repetition artifacts. Sentence-aware chunking and improved answer formatting are potential future improvements.

### 2. Refund agent

Implemented a simulated refund agent using the supplied local LLM mock and project data.

The agent:
- Reads order and customer information from the local SQLite database.
- Retrieves relevant refund policy information.
- Checks order status and delivery time.
- Applies the automatic refund limit of INR 1,000.
- Escalates refunds above INR 1,000 for human review.
- Records simulated refund actions in a CSV ledger.
- Uses checkpoints and trace logs to support crash recovery and debugging.
- Applies a maximum-step guard to prevent infinite agent loops.

Scenario results:

| Scenario | Outcome | Refund amount |
|---|---|---:|
| S1 | Refund issued | INR 128.25 |
| S2 | Refund issued | INR 135.00 |
| S3 | Escalated for human review | INR 1,999.00 |
| S4 | Denied: delivery within 45 minutes | INR 0 |
| S5 | Denied: order not completed | INR 0 |
| S6 | Order not found | INR 0 |
| S7 | Safely stopped by maximum-step guard | INR 0 |

All refund actions are simulated. No real payment or financial transaction is performed. The human-review amount is not an issued refund.

### 3. ReAct vs Plan-and-Execute comparison

Compared the ReAct refund agent outcomes with a deterministic, read-only Plan-and-Execute simulation across seven scenarios.

- Scenarios evaluated: 7
- Matching outcome statuses: 7
- Status match rate: 100%
- Average ReAct loop turns: 4.14
- Average Plan-and-Execute total steps: 5.00

The Plan-and-Execute implementation is a deterministic simulation, not a second LLM agent with equivalent token accounting. Therefore, the step counts should not be interpreted as a direct performance or cost comparison.

### 4. Part D output files

Policy RAG:
- `outputs/policy_chunks_overlap.csv`
- `outputs/policy_chunks_no_overlap.csv`
- `outputs/policy_retrieval_results.csv`
- `outputs/policy_eval_overlap.csv`
- `outputs/policy_eval_no_overlap.csv`
- `outputs/policy_chunk_comparison.csv`

Refund agent:
- `outputs/refund_agent_results.csv`
- `outputs/simulated_refund_ledger.csv`
- `outputs/refund_agent_checkpoint.json`
- `outputs/refund_agent_trace.jsonl`

Architecture comparison:
- `outputs/agent_architecture_comparison.csv`
- `outputs/agent_architecture_summary.csv`

Part D source scripts:
- `src/part_d_policy.py`
- `src/part_d_refund_agent.py`
- `src/part_d_agent_comparison.py`


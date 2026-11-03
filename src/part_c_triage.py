
from pathlib import Path
import json
import random
import re
import sys
import time

import pandas as pd
from sklearn.metrics import confusion_matrix

# Make the project root importable so we can use the supplied mock_llm.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

ZERO_SHOT_BUDGET = 6000
FINAL_BUDGET = 100000

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]

ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only, using exactly this structure:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not add Markdown, code fences, or any text outside the JSON.
Use the customer's ticket text to determine the best category.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only, using exactly this structure:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Do not add Markdown, code fences, or any text outside the JSON.
Use the customer's ticket text to determine the best category.

EXAMPLES:
Ticket: "I was charged twice and need my bill corrected."
Answer: {"category": "billing", "reason": "Duplicate charge"}

Ticket: "Someone made a transaction on my card without permission."
Answer: {"category": "fraud", "reason": "Unauthorized transaction"}

Ticket: "The application crashes when I try to pay."
Answer: {"category": "technical", "reason": "Application error"}

Ticket: "My parcel has not arrived at the promised time."
Answer: {"category": "delivery", "reason": "Late delivery"}
""".strip()


# ---------------------------------------------------------
# 2. Load data and redact personally identifiable information
# ---------------------------------------------------------

def load_data():
    if not RAW_TICKETS.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_TICKETS}")

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing Part B output: {SEGMENTS_FILE}")

    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)

    required_ticket_columns = {
        "ticket_id", "customer_id", "text", "true_category"
    }
    missing = required_ticket_columns - set(tickets.columns)
    if missing:
        raise ValueError(f"tickets.csv is missing columns: {sorted(missing)}")

    required_segment_columns = {
        "customer_id", "segment_name", "is_premium"
    }
    missing = required_segment_columns - set(segments.columns)
    if missing:
        raise ValueError(
            f"customer_segments.csv is missing columns: {sorted(missing)}"
        )

    # Keep the original ticket order.
    tickets = tickets.copy()
    tickets["text"] = tickets["text"].fillna("").astype(str)
    tickets["true_category"] = (
        tickets["true_category"].astype(str).str.strip().str.lower()
    )

    return tickets, segments


def redact_text(text):
    """Redact in the required order and return the text plus match counts."""
    text = str(text or "")
    counts = {"emails": 0, "phones": 0, "order_ids": 0}

    def replace_email(match):
        counts["emails"] += 1
        return "[EMAIL]"

    def replace_phone(match):
        counts["phones"] += 1
        return "[PHONE]"

    def replace_order(match):
        counts["order_ids"] += 1
        return "[ORDER_ID]"

    text = EMAIL_RE.sub(replace_email, text)
    text = PHONE_RE.sub(replace_phone, text)
    text = ORDER_RE.sub(replace_order, text)

    # Safety check: no original PII pattern may remain.
    assert EMAIL_RE.search(text) is None, "Email pattern survived redaction"
    assert PHONE_RE.search(text) is None, "Phone pattern survived redaction"
    assert ORDER_RE.search(text) is None, "Order ID survived redaction"

    return text, counts


def prepare_tickets(tickets):
    prepared = []
    totals = {"emails": 0, "phones": 0, "order_ids": 0}

    for _, row in tickets.iterrows():
        redacted, counts = redact_text(row["text"])

        for key in totals:
            totals[key] += counts[key]

        original_text = row["text"]

        if not original_text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            status = "pending"

        prepared.append({
            "ticket_id": str(row["ticket_id"]),
            "customer_id": str(row["customer_id"]),
            "text": original_text,
            "redacted_text": redacted,
            "true_category": str(row["true_category"]),
            "initial_status": status,
        })

    totals["total"] = (
        totals["emails"] + totals["phones"] + totals["order_ids"]
    )

    print("\n--- PII redaction audit ---")
    print(f"Email matches redacted: {totals['emails']}")
    print(f"Phone matches redacted: {totals['phones']}")
    print(f"Order IDs redacted:     {totals['order_ids']}")
    print(f"Total matches redacted: {totals['total']}")

    remaining = {
        "emails": sum(bool(EMAIL_RE.search(t["redacted_text"])) for t in prepared),
        "phones": sum(bool(PHONE_RE.search(t["redacted_text"])) for t in prepared),
        "order_ids": sum(bool(ORDER_RE.search(t["redacted_text"])) for t in prepared),
    }
    print(f"Remaining PII-pattern matches: {sum(remaining.values())}")
    assert sum(remaining.values()) == 0

    print("\n--- Input guardrail audit ---")
    print("Empty tickets:", sum(t["initial_status"] == "skipped_empty" for t in prepared))
    print("Long tickets:", sum(t["initial_status"] == "flagged_long" for t in prepared))
    print("Eligible tickets:", sum(t["initial_status"] == "pending" for t in prepared))

    return prepared, totals


# ---------------------------------------------------------
# 3. Validate prompts and model output
# ---------------------------------------------------------

def validate_prompt(prompt, few_shot):
    assert "CLASSIFY" in prompt
    for category in CATEGORIES:
        assert category in prompt

    assert '{"category": "...", "reason": "..."}' in prompt

    if few_shot:
        assert "EXAMPLES:" in prompt
        assert '"category": "fraud"' in prompt
    else:
        assert "EXAMPLES:" not in prompt


def validate_no_pii_in_messages(messages):
    for message in messages:
        content = str(message.get("content") or "")
        assert EMAIL_RE.search(content) is None, "Email pattern in model messages"
        assert PHONE_RE.search(content) is None, "Phone pattern in model messages"
        assert ORDER_RE.search(content) is None, "Order ID pattern in model messages"


def validate_reply(reply_text):
    """Return (status, category); never guess an ambiguous category."""
    try:
        data = json.loads(reply_text)
        if isinstance(data, dict):
            category = data.get("category")
            if isinstance(category, str) and category.lower() in CATEGORIES:
                return "classified_json", category.lower()
    except (json.JSONDecodeError, TypeError):
        pass

    # Find category words even inside hyphenated words such as billing-related.
    words = re.findall(r"[a-z]+", str(reply_text).lower())
    found = set(words) & CATEGORIES

    if len(found) == 1:
        return "classified_keyword", next(iter(found))

    return "flagged_invalid_output", ""


# ---------------------------------------------------------
# 4. Retry handling and one full triage run
# ---------------------------------------------------------

def call_model_with_retries(messages, ticket_id, token_state, token_budget, retry_log):
    """
    Check the token budget before every attempt.
    Retry transient errors up to three times after the initial attempt.
    """
    for attempt in range(MAX_RETRIES + 1):
        if token_state["total"] >= token_budget:
            return None, "budget_exhausted"

        try:
            response = mock_llm.chat(messages)
            return response, "ok"

        except mock_llm.TransientLLMError:
            if attempt >= MAX_RETRIES:
                return None, "permanent_error"

            wait_seconds = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            log_line = (
                f"ticket_id={ticket_id} "
                f"retry={attempt + 1} "
                f"wait={wait_seconds:.4f}s"
            )
            print("RETRY:", log_line)
            retry_log.append(log_line)
            time.sleep(wait_seconds)

    return None, "permanent_error"


def route_ticket(status, category, is_premium):
    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "permanent_error",
    }

    if status in human_review_statuses:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


def is_premium_value(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def run_triage(prepared, segments, prompt, few_shot, token_budget, run_name):
    validate_prompt(prompt, few_shot)

    # Reset the supplied mock immediately before each complete run.
    mock_llm.reset()

    token_state = {"total": 0}
    retry_log = []
    results = []

    segment_lookup = {}
    for _, row in segments.iterrows():
        segment_lookup[str(row["customer_id"])] = {
            "segment_name": str(row["segment_name"]),
            "is_premium": is_premium_value(row["is_premium"]),
        }

    print(f"\n{'=' * 60}")
    print(f"RUN: {run_name}")
    print(f"Token budget: {token_budget}")
    print(f"{'=' * 60}")

    for ticket in prepared:
        status = ticket["initial_status"]
        category = ""
        tokens_used = 0

        if status == "pending":
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": ticket["redacted_text"]},
            ]

            # User message contains only the redacted ticket text.
            assert messages[-1]["content"] == ticket["redacted_text"]
            validate_no_pii_in_messages(messages)

            response, call_status = call_model_with_retries(
                messages=messages,
                ticket_id=ticket["ticket_id"],
                token_state=token_state,
                token_budget=token_budget,
                retry_log=retry_log,
            )

            if call_status == "budget_exhausted":
                status = "budget_exhausted"

            elif call_status == "permanent_error":
                status = "permanent_error"

            else:
                tokens_used = int(response.usage.total_tokens)
                token_state["total"] += tokens_used

                reply_text = response.choices[0].message.content or ""
                status, category = validate_reply(reply_text)

        customer_segment = segment_lookup.get(
            ticket["customer_id"],
            {"segment_name": "", "is_premium": False},
        )

        segment_name = customer_segment["segment_name"]
        is_premium = customer_segment["is_premium"]

        route, priority = route_ticket(status, category, is_premium)

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "true_category": ticket["true_category"],
        })

    results_df = pd.DataFrame(results)

    # Persist retry logs, including the run name for easy comparison.
    return results_df, token_state["total"], retry_log


# ---------------------------------------------------------
# 5. Experiment metrics and reporting
# ---------------------------------------------------------

def report_experiment(results, total_tokens, run_name):
    classified_statuses = {"classified_json", "classified_keyword"}
    classified = results[results["status"].isin(classified_statuses)].copy()

    correct = (
        classified["category"].str.lower()
        == classified["true_category"].str.lower()
    ).sum()

    classified_count = len(classified)
    accuracy = correct / classified_count if classified_count else 0.0
    tokens_per_ticket = (
        total_tokens / classified_count if classified_count else 0.0
    )

    print(f"\n--- {run_name} metrics ---")
    print(f"Tickets in input:             {len(results)}")
    print(f"Tickets classified:           {classified_count}")
    print(f"Correct classifications:      {correct}")
    print(f"Accuracy on classified:       {accuracy:.1%}")
    print(f"Total response tokens:        {total_tokens}")
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")
    print("\nStatus counts:")
    print(results["status"].value_counts().sort_index().to_string())

    return {
        "run_name": run_name,
        "tickets_classified": classified_count,
        "correct": int(correct),
        "accuracy": accuracy,
        "total_tokens": int(total_tokens),
        "tokens_per_classified_ticket": tokens_per_ticket,
    }


def report_confusion_matrix(results):
    classified_statuses = {"classified_json", "classified_keyword"}
    classified = results[results["status"].isin(classified_statuses)]

    matrix = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )

    print("\n--- Zero-shot confusion table ---")
    print(matrix.to_string())

    return matrix


def save_results(results, path):
    output = results[OUTPUT_COLUMNS].copy()
    output.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    print(f"Rows written: {len(output)}")
    print("Columns:", ", ".join(output.columns))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets, segments = load_data()
    prepared, pii_totals = prepare_tickets(tickets)

    # Fixed seed makes retry wait times reproducible.
    random.seed(42)

    # Run 1: zero-shot evaluation, with a large enough budget to finish.
    zero_results, zero_tokens, zero_retries = run_triage(
        prepared=prepared,
        segments=segments,
        prompt=ZERO_SHOT_PROMPT,
        few_shot=False,
        token_budget=100000,
        run_name="zero-shot evaluation",
    )
    zero_metrics = report_experiment(
        zero_results, zero_tokens, "Zero-shot"
    )
    report_confusion_matrix(zero_results)

    # Run 2: few-shot evaluation, with a large enough budget to finish.
    few_results, few_tokens, few_retries = run_triage(
        prepared=prepared,
        segments=segments,
        prompt=FEW_SHOT_PROMPT,
        few_shot=True,
        token_budget=100000,
        run_name="few-shot evaluation",
    )
    few_metrics = report_experiment(
        few_results, few_tokens, "Few-shot"
    )

    # Run 3: zero-shot budget demonstration.
    budget_results, budget_tokens, budget_retries = run_triage(
        prepared=prepared,
        segments=segments,
        prompt=ZERO_SHOT_PROMPT,
        few_shot=False,
        token_budget=ZERO_SHOT_BUDGET,
        run_name="zero-shot budget demonstration",
    )
    report_experiment(
        budget_results, budget_tokens, "Zero-shot budget demo"
    )

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    save_results(budget_results, budget_path)

    exhausted_count = int(
        (budget_results["status"] == "budget_exhausted").sum()
    )
    print(f"Tickets stopped by the token budget: {exhausted_count}")

    # Run 4: final deliverable, few-shot, with the required large budget.
    final_results, final_tokens, final_retries = run_triage(
        prepared=prepared,
        segments=segments,
        prompt=FEW_SHOT_PROMPT,
        few_shot=True,
        token_budget=FINAL_BUDGET,
        run_name="final few-shot triage",
    )
    final_metrics = report_experiment(
        final_results, final_tokens, "Final few-shot triage"
    )

    final_path = OUTPUT_DIR / "triage_results.csv"
    save_results(final_results, final_path)

    # All retry events from all runs are saved for auditability.
    retry_path = OUTPUT_DIR / "triage_retry.log"
    with retry_path.open("w", encoding="utf-8") as log_file:
        for run_name, lines in [
            ("zero-shot evaluation", zero_retries),
            ("few-shot evaluation", few_retries),
            ("budget demonstration", budget_retries),
            ("final few-shot triage", final_retries),
        ]:
            log_file.write(f"\n[{run_name}]\n")
            for line in lines:
                log_file.write(line + "\n")

    print(f"\nSaved retry log: {retry_path}")
    print("\n--- Final run retry audit ---")
    print(f"Final few-shot retry events: {len(final_retries)}")
    print(f"All runs' retry events: "
          f"{len(zero_retries) + len(few_retries) + len(budget_retries) + len(final_retries)}")

    # Compact comparison.
    comparison = pd.DataFrame([zero_metrics, few_metrics])
    print("\n--- Zero-shot vs few-shot comparison ---")
    print(comparison.to_string(index=False))

    print("\n--- Final status counts ---")
    print(final_results["status"].value_counts().sort_index().to_string())

    # Acceptance checks that do not assume the model's labels are correct.
    assert len(final_results) == len(tickets), "Final output row count mismatch"
    assert list(final_results[OUTPUT_COLUMNS].columns) == OUTPUT_COLUMNS
    assert (final_results["status"] == "skipped_empty").sum() == 4
    assert (final_results["status"] == "flagged_long").sum() == 4
    assert exhausted_count > 0, (
        "No budget exhaustion occurred. Lower ZERO_SHOT_BUDGET and rerun."
    )

    print("\nPart C script completed. Review the metrics and CSV files above.")


if __name__ == "__main__":
    main()

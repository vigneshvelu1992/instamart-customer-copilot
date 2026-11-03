
from pathlib import Path
import sys
import json
import csv
import sqlite3
import time
import re

import pandas as pd

# ---------------------------------------------------------
# 1. Project paths and imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SCENARIOS_PATH = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

LEDGER_PATH = OUTPUT_DIR / "simulated_refund_ledger.csv"
RESULTS_PATH = OUTPUT_DIR / "refund_agent_results.csv"
TRACE_PATH = OUTPUT_DIR / "refund_agent_trace.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / "refund_agent_checkpoint.json"

MAX_STEPS = 6
MAX_RETRIES = 4
MAX_REFUND_INR = 1000.0
CRASH_SCENARIO_INDEX = 2  # Third scenario, normally S3


# ---------------------------------------------------------
# 2. General helpers
# ---------------------------------------------------------

def ensure_paths():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {DB_PATH}")

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIOS_PATH}"
        )

    if not POLICIES_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICIES_DIR}"
        )


def append_trace(event):
    """Append one JSON record to the execution trace."""
    with TRACE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_policy_text():
    """Load policy documents for the policy-search tool."""
    documents = {}

    for path in sorted(POLICIES_DIR.glob("*.txt")):
        documents[path.stem] = path.read_text(encoding="utf-8")

    if not documents:
        raise FileNotFoundError(
            f"No .txt policy files found in {POLICIES_DIR}"
        )

    return documents


POLICY_DOCUMENTS = None


def search_policy(query):
    """Simple offline policy search using word overlap."""
    global POLICY_DOCUMENTS

    if POLICY_DOCUMENTS is None:
        POLICY_DOCUMENTS = load_policy_text()

    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []

    for name, text in POLICY_DOCUMENTS.items():
        text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
        score = len(query_words & text_words)
        ranked.append((score, name, text))

    ranked.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, name, text in ranked[:3]:
        if score > 0:
            results.append({
                "policy": name,
                "score": score,
                "excerpt": text[:1800]
            })

    if not results:
        return {
            "found": False,
            "message": "No matching policy text found."
        }

    return {"found": True, "results": results}


# ---------------------------------------------------------
# 3. Read-only order lookup
# ---------------------------------------------------------

def get_order(order_id):
    """Fetch an order from SQLite; this tool does not change the DB."""
    if not re.fullmatch(r"ORD-\d{6}", str(order_id)):
        return {"error": "invalid_order_id"}

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT order_id, customer_id, status,
                   order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if row is None:
        return {"error": "order_not_found", "order_id": order_id}

    order = dict(row)

    try:
        order["order_amount"] = float(order["order_amount"])
    except (TypeError, ValueError):
        order["order_amount"] = None

    try:
        order["delivery_minutes"] = (
            float(order["delivery_minutes"])
            if order["delivery_minutes"] is not None
            else None
        )
    except (TypeError, ValueError):
        order["delivery_minutes"] = None

    order["status"] = str(order["status"]).strip().lower()

    return order


# ---------------------------------------------------------
# 4. Refund calculation and simulated action tools
# ---------------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    """
    Conservative demo rule:
    - Refund only completed orders.
    - Delivery must be more than 45 minutes.
    - Refund is capped at the original order amount.
    - Refunds over INR 1,000 require human review.
    """
    if order_amount is None or delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "decision": "human_review",
            "reason": "Missing amount or delivery-time data."
        }

    amount = float(order_amount)
    minutes = float(delivery_minutes)

    if amount <= 0:
        return {
            "refund_inr": 0.0,
            "decision": "denied",
            "reason": "Order amount is not valid."
        }

    if minutes <= 45:
        return {
            "refund_inr": 0.0,
            "decision": "denied",
            "reason": "Delivery was within 45 minutes."
        }

    refund = round(min(amount, amount), 2)

    if refund > MAX_REFUND_INR:
        return {
            "refund_inr": refund,
            "decision": "human_review",
            "reason": "Refund exceeds the INR 1,000 automatic limit."
        }

    return {
        "refund_inr": refund,
        "decision": "eligible",
        "reason": "Completed order exceeded the 45-minute threshold."
    }


def read_ledger():
    if not LEDGER_PATH.exists():
        return []

    with LEDGER_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_ledger(rows):
    fields = [
        "order_id",
        "refund_inr",
        "action",
        "scenario_id",
        "timestamp"
    ]

    with LEDGER_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def issue_refund(order_id, refund_inr, scenario_id):
    """
    Simulated refund only: writes to a CSV ledger.
    It never sends money or modifies the SQLite database.
    """
    amount = round(float(refund_inr), 2)

    if amount <= 0:
        return {"error": "invalid_refund_amount"}

    if amount > MAX_REFUND_INR:
        return {"error": "refund_exceeds_automatic_limit"}

    order = get_order(order_id)

    if "error" in order:
        return {"error": "order_not_found"}

    if order["status"] != "completed":
        return {"error": "order_not_completed"}

    if order["order_amount"] is None or amount > order["order_amount"]:
        return {"error": "refund_exceeds_order_amount"}

    rows = read_ledger()

    # Idempotency: do not issue the same simulated refund twice.
    for row in rows:
        if (
            row["order_id"] == order_id
            and row["action"] == "refund_issued"
        ):
            return {
                "already_processed": True,
                "order_id": order_id,
                "refund_inr": float(row["refund_inr"])
            }

    rows.append({
        "order_id": order_id,
        "refund_inr": f"{amount:.2f}",
        "action": "refund_issued",
        "scenario_id": scenario_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    write_ledger(rows)

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": amount,
        "simulated": True
    }


def escalate_to_human(order_id, refund_inr, scenario_id, reason):
    """Record a human-review request without issuing a refund."""
    rows = read_ledger()

    rows.append({
        "order_id": order_id,
        "refund_inr": f"{float(refund_inr):.2f}",
        "action": "human_review",
        "scenario_id": scenario_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    write_ledger(rows)

    return {
        "escalated": True,
        "order_id": order_id,
        "refund_inr": float(refund_inr),
        "reason": reason
    }


# ---------------------------------------------------------
# 5. Tool definitions passed to mock_llm
# ---------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order in the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the local policy documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility and amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {
                        "type": ["number", "null"]
                    },
                    "delivery_minutes": {
                        "type": ["number", "null"]
                    }
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an eligible simulated refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a refund request for human review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


# ---------------------------------------------------------
# 6. Retry wrapper for transient mock-model errors
# ---------------------------------------------------------

def call_mock_with_retry(messages):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return mock_llm.chat(
                messages,
                tools=TOOL_DEFINITIONS
            )
        except mock_llm.TransientLLMError as error:
            last_error = error

            append_trace({
                "event": "transient_error",
                "attempt": attempt,
                "error": str(error)
            })

            if attempt < MAX_RETRIES:
                time.sleep(0.2 * attempt)

    raise RuntimeError(
        f"Mock LLM failed after {MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------
# 7. Tool execution with local safety checks
# ---------------------------------------------------------

def execute_tool(name, arguments, scenario_id, last_order):
    if name == "get_order":
        order_id = arguments.get("order_id", "")
        result = get_order(order_id)
        return result, result if "error" not in result else None

    if name == "search_policy":
        query = str(arguments.get("query", "refund policy"))
        return search_policy(query), last_order

    if name == "calculate_refund":
        # Use verified database order data, not model-supplied values.
        if not last_order:
            return {"error": "order_lookup_required"}, last_order

        if last_order.get("status") != "completed":
            return {
                "refund_inr": 0.0,
                "decision": "denied",
                "reason": "Only completed orders can be refunded."
            }, last_order

        result = calculate_refund(
            last_order.get("order_amount"),
            last_order.get("delivery_minutes")
        )
        return result, last_order

    if name == "issue_refund":
        if not last_order:
            return {"error": "order_lookup_required"}, last_order

        order_id = str(arguments.get("order_id", ""))
        requested_amount = float(arguments.get("refund_inr", 0))

        # Never allow the model to choose a different order.
        if order_id != last_order.get("order_id"):
            return {"error": "order_id_mismatch"}, last_order

        decision = calculate_refund(
            last_order.get("order_amount"),
            last_order.get("delivery_minutes")
        )

        if decision["decision"] != "eligible":
            return {
                "error": "refund_not_eligible",
                "reason": decision["reason"]
            }, last_order

        # Amount must exactly match the verified calculation.
        if round(requested_amount, 2) != round(
            float(decision["refund_inr"]), 2
        ):
            return {"error": "refund_amount_mismatch"}, last_order

        result = issue_refund(
            order_id,
            requested_amount,
            scenario_id
        )
        return result, last_order

    if name == "escalate_to_human":
        if not last_order:
            return {"error": "order_lookup_required"}, last_order

        order_id = str(arguments.get("order_id", ""))

        if order_id != last_order.get("order_id"):
            return {"error": "order_id_mismatch"}, last_order

        decision = calculate_refund(
            last_order.get("order_amount"),
            last_order.get("delivery_minutes")
        )

        if decision["decision"] != "human_review":
            return {
                "error": "human_escalation_not_required",
                "reason": decision["reason"]
            }, last_order

        result = escalate_to_human(
            order_id,
            decision["refund_inr"],
            scenario_id,
            decision["reason"]
        )
        return result, last_order

    return {"error": f"unknown_tool:{name}"}, last_order


# ---------------------------------------------------------
# 8. Scenario execution
# ---------------------------------------------------------

def run_scenario(scenario, scenario_index):
    scenario_id = str(scenario["scenario_id"])
    user_text = str(scenario["message"])

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: Help with refund requests using tools. "
                "Look up the order, search the policy, calculate "
                "eligibility, and use safe tools. Never invent data."
            )
        },
        {"role": "user", "content": user_text}
    ]

    last_order = None
    tool_steps = 0

    # The supplied mock intentionally repeats a tool call for [LOOP].
    # Bound the test explicitly so it cannot enter an infinite loop.
    if "[LOOP]" in user_text:
        for step in range(1, MAX_STEPS + 1):
            response = call_mock_with_retry(messages)
            tool_calls = getattr(
                response.choices[0].message,
                "tool_calls",
                None
            )

            append_trace({
                "scenario_id": scenario_id,
                "event": "loop_guard_test",
                "step": step,
                "tool_call_returned": bool(tool_calls)
            })

            if not tool_calls:
                content = response.choices[0].message.content or ""
                return {
                    "scenario_id": scenario_id,
                    "status": "completed",
                    "order_id": "",
                    "refund_inr": 0.0,
                    "agent_response": str(content),
                    "tool_steps": step
                }

        return {
            "scenario_id": scenario_id,
            "status": "step_limit",
            "order_id": "",
            "refund_inr": 0.0,
            "agent_response": "SAFE_STOP: maximum agent steps reached.",
            "tool_steps": MAX_STEPS
        }

    for step in range(1, MAX_STEPS + 1):
        tool_steps = step

        try:
            response = call_mock_with_retry(messages)
        except Exception as error:
            return {
                "scenario_id": scenario_id,
                "status": "model_error",
                "order_id": (
                    last_order.get("order_id") if last_order else ""
                ),
                "refund_inr": 0.0,
                "agent_response": f"SAFE_STOP: {error}",
                "tool_steps": tool_steps
            }

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            content = message.content or ""
            status = "completed"

            if str(content).startswith("NOT_FOUND:"):
                status = "order_not_found"
            elif str(content).startswith("DENIED:"):
                status = "denied"
            elif str(content).startswith("ESCALATE:"):
                status = "human_review"
            elif str(content).startswith("REFUND:"):
                status = "refund_response"
            elif str(content).startswith("NEEDS_INFO:"):
                status = "needs_info"

            return {
                "scenario_id": scenario_id,
                "status": status,
                "order_id": (
                    last_order.get("order_id") if last_order else ""
                ),
                "refund_inr": 0.0,
                "agent_response": str(content),
                "tool_steps": tool_steps
            }

        # This mock returns flat attributes: tc.name, tc.arguments, tc.id.
        for tool_call in tool_calls:
            name = getattr(tool_call, "name", None)
            raw_arguments = getattr(tool_call, "arguments", "{}")
            tool_id = getattr(tool_call, "id", "")

            if not name:
                return {
                    "scenario_id": scenario_id,
                    "status": "invalid_tool_call",
                    "order_id": "",
                    "refund_inr": 0.0,
                    "agent_response": (
                        "SAFE_STOP: tool call had no valid tool name."
                    ),
                    "tool_steps": tool_steps
                }

            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object.")
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                return {
                    "scenario_id": scenario_id,
                    "status": "invalid_tool_arguments",
                    "order_id": "",
                    "refund_inr": 0.0,
                    "agent_response": f"SAFE_STOP: {error}",
                    "tool_steps": tool_steps
                }

            append_trace({
                "scenario_id": scenario_id,
                "event": "tool_call",
                "step": step,
                "tool": name,
                "arguments": arguments
            })

            # For an unknown order, stop cleanly with a clear result.
            if name == "get_order":
                verified_order = get_order(
                    str(arguments.get("order_id", ""))
                )

                if verified_order.get("error") == "order_not_found":
                    return {
                        "scenario_id": scenario_id,
                        "status": "order_not_found",
                        "order_id": str(arguments.get("order_id", "")),
                        "refund_inr": 0.0,
                        "agent_response": "NOT_FOUND: no such order.",
                        "tool_steps": tool_steps
                    }

                if verified_order.get("error"):
                    return {
                        "scenario_id": scenario_id,
                        "status": "invalid_order_id",
                        "order_id": str(arguments.get("order_id", "")),
                        "refund_inr": 0.0,
                        "agent_response": "SAFE_STOP: invalid order ID.",
                        "tool_steps": tool_steps
                    }

            result, last_order = execute_tool(
                name,
                arguments,
                scenario_id,
                last_order
            )

            append_trace({
                "scenario_id": scenario_id,
                "event": "tool_result",
                "step": step,
                "tool": name,
                "result": result
            })

            # The mock uses tool messages to determine the next action.
            messages.append({
                "role": "tool",
                "tool_call_id": str(tool_id),
                "name": name,
                "content": json.dumps(result, ensure_ascii=False)
            })

            # Stop on any safety validation failure.
            if isinstance(result, dict) and result.get("error"):
                return {
                    "scenario_id": scenario_id,
                    "status": "safe_stop",
                    "order_id": (
                        last_order.get("order_id") if last_order else ""
                    ),
                    "refund_inr": 0.0,
                    "agent_response": (
                        "SAFE_STOP: " + str(result.get("error"))
                    ),
                    "tool_steps": tool_steps
                }

    return {
        "scenario_id": scenario_id,
        "status": "step_limit",
        "order_id": last_order.get("order_id") if last_order else "",
        "refund_inr": 0.0,
        "agent_response": "SAFE_STOP: maximum agent steps reached.",
        "tool_steps": MAX_STEPS
    }


# ---------------------------------------------------------
# 9. Checkpoint / resume
# ---------------------------------------------------------

def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return {
            "next_index": 0,
            "completed_results": [],
            "crash_simulated": False
        }

    try:
        checkpoint = json.loads(
            CHECKPOINT_PATH.read_text(encoding="utf-8")
        )
        checkpoint.setdefault("next_index", 0)
        checkpoint.setdefault("completed_results", [])
        checkpoint.setdefault("crash_simulated", False)
        return checkpoint
    except (json.JSONDecodeError, OSError):
        return {
            "next_index": 0,
            "completed_results": [],
            "crash_simulated": False
        }


def save_checkpoint(checkpoint):
    CHECKPOINT_PATH.write_text(
        json.dumps(checkpoint, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ---------------------------------------------------------
# 10. Load scenarios and run the agent
# ---------------------------------------------------------

def load_scenarios():
    frame = pd.read_csv(SCENARIOS_PATH)

    # Accept common column-name variants in the supplied CSV.
    lower_to_original = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    id_candidates = [
        "scenario_id", "scenario", "id"
    ]
    message_candidates = [
        "message", "customer_message", "query", "request", "text"
    ]

    id_column = next(
        (lower_to_original[name] for name in id_candidates
         if name in lower_to_original),
        None
    )
    message_column = next(
        (lower_to_original[name] for name in message_candidates
         if name in lower_to_original),
        None
    )

    if message_column is None:
        raise ValueError(
            "Could not identify the message column in agent_scenarios.csv. "
            f"Found columns: {list(frame.columns)}"
        )

    if id_column is None:
        frame["scenario_id"] = [
            f"S{i + 1}" for i in range(len(frame))
        ]
        id_column = "scenario_id"

    frame = frame.rename(columns={
        id_column: "scenario_id",
        message_column: "message"
    })

    return frame.fillna("").to_dict(orient="records")


def main():
    ensure_paths()

    scenarios = load_scenarios()
    checkpoint = load_checkpoint()

    completed_results = checkpoint["completed_results"]
    start_index = int(checkpoint["next_index"])

    # Reset the trace for a fresh run, but preserve it when resuming.
    if start_index == 0 and TRACE_PATH.exists():
        TRACE_PATH.unlink()

    append_trace({
        "event": "run_started",
        "scenario_count": len(scenarios),
        "resume_index": start_index
    })

    # Simulate one crash, save the checkpoint, and resume safely.
    if (
        start_index <= CRASH_SCENARIO_INDEX
        and CRASH_SCENARIO_INDEX < len(scenarios)
        and not checkpoint["crash_simulated"]
    ):
        checkpoint["crash_simulated"] = True
        save_checkpoint(checkpoint)

        append_trace({
            "event": "simulated_crash",
            "scenario_index": CRASH_SCENARIO_INDEX,
            "checkpoint_saved": True
        })

        print(
            "Simulated crash recorded. "
            "Checkpoint saved; resuming from the saved position..."
        )

        checkpoint = load_checkpoint()
        completed_results = checkpoint["completed_results"]
        start_index = int(checkpoint["next_index"])

    for index in range(start_index, len(scenarios)):
        scenario = scenarios[index]

        print(
            f"Processing {index + 1}/{len(scenarios)}: "
            f"{scenario['scenario_id']}"
        )

        result = run_scenario(scenario, index)
        completed_results.append(result)

        checkpoint["completed_results"] = completed_results
        checkpoint["next_index"] = index + 1
        save_checkpoint(checkpoint)

    results_frame = pd.DataFrame(completed_results)
    results_frame.to_csv(RESULTS_PATH, index=False)

    append_trace({
        "event": "run_finished",
        "processed_scenarios": len(completed_results),
        "results_path": str(RESULTS_PATH)
    })

    print("\nPart D refund agent finished.")
    print(f"Scenarios processed: {len(completed_results)}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Ledger: {LEDGER_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Trace: {TRACE_PATH}")

    if not results_frame.empty:
        print("\nScenario status summary:")
        print(results_frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()

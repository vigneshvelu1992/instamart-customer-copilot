from pathlib import Path
from datetime import datetime, timezone
import json
import re
import sqlite3
import sys
import time

import pandas as pd

# ---------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm

from part_d_policy import (
    load_policy_documents,
    build_chunks,
    PolicyRetriever,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
POLICY_DIR = ROOT / "data" / "policies"

OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

AGENT_RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RESULTS_PATH = OUTPUT_DIR / "refund_agent_results.csv"
TRACE_PATH = OUTPUT_DIR / "refund_agent_trace.jsonl"
SAFETY_DEMOS_PATH = OUTPUT_DIR / "agent_safety_demos.jsonl"

MAX_ITERATIONS = 6
MAX_REFUND_INR = 1000.00

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

def ensure_paths():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_id(run_id):
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(run_id))
    if not cleaned:
        raise ValueError("run_id cannot be empty")
    return cleaned


def checkpoint_path(run_id):
    return STATE_DIR / f"{safe_run_id(run_id)}.json"


# ---------------------------------------------------------
# Policy retrieval
# ---------------------------------------------------------

_POLICY_RETRIEVER = None


def get_policy_retriever():
    global _POLICY_RETRIEVER

    if _POLICY_RETRIEVER is None:
        documents = load_policy_documents()

        chunks = build_chunks(
            documents,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        _POLICY_RETRIEVER = PolicyRetriever(chunks)

    return _POLICY_RETRIEVER


def search_policy(query):
    """Retrieve the three most relevant policy chunks."""

    retriever = get_policy_retriever()

    retrieved = retriever.retrieve(query, top_k=3)

    results = [
        {
            "source": item["source"],
            "score": float(item["score"]),
            "text": item["text"],
        }
        for item in retrieved
    ]

    return {"results": results}


# ---------------------------------------------------------
# Tool 1: Get order
# ---------------------------------------------------------

def get_order(order_id):
    """Read one order from the SQLite database."""

    order_id = str(order_id).strip()

    if not re.fullmatch(r"ORD-\d{6}", order_id):
        return {"error": "Invalid order ID format"}

    if not DB_PATH.exists():
        return {"error": "Order database not found"}

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    finally:
        connection.close()

    if row is None:
        return {"error": "Order not found"}

    return {
        "order_id": str(row["order_id"]),
        "status": str(row["status"]).strip().lower(),
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            None
            if pd.isna(row["delivery_minutes"])
            else float(row["delivery_minutes"])
        ),
    }


# ---------------------------------------------------------
# Tool 3: Calculate refund
# ---------------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    """
    Refund rules:
    - Delivery time must be greater than 45 minutes.
    - Eligible completed orders receive 100% of order amount.
    - Refunds above INR 1,000 require human escalation.
    """

    amount = float(order_amount)
    delivery = float(delivery_minutes)

    if amount < 0 or delivery < 0:
        return {
            "refund_inr": 0.0,
            "decision": "denied",
            "reason": "Invalid order amount or delivery time.",
        }

    if delivery <= 45:
        return {
            "refund_inr": 0.0,
            "decision": "denied",
            "reason": "Delivery time was not greater than 45 minutes.",
        }

    if amount > MAX_REFUND_INR:
        return {
            "refund_inr": round(amount, 2),
            "decision": "human_review",
            "reason": "Eligible refund exceeds INR 1,000.",
        }

    return {
        "refund_inr": round(amount, 2),
        "decision": "eligible",
        "reason": "Order qualifies for a full refund.",
    }


# ---------------------------------------------------------
# Refund ledger helpers
# ---------------------------------------------------------

def read_ledger_records():
    if not LEDGER_PATH.exists():
        return []

    records = []

    with LEDGER_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def ledger_entry_exists(order_id, tool_name):
    return any(
        row.get("order_id") == order_id
        and row.get("tool") == tool_name
        for row in read_ledger_records()
    )


def validate_refund_action(order_id, refund_inr):
    order = get_order(order_id)

    if order.get("error"):
        return None, order["error"]

    if order["status"] != "completed":
        return None, "Only completed orders can be refunded."

    amount = float(refund_inr)

    if amount <= 0:
        return None, "Refund amount must be greater than zero."

    if amount > order["order_amount"] + 0.01:
        return None, "Refund exceeds the original order amount."

    if amount > MAX_REFUND_INR:
        return None, "Refund exceeds the automatic refund limit."

    decision = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"],
    )

    if decision["decision"] != "eligible":
        return None, decision["reason"]

    if abs(amount - decision["refund_inr"]) > 0.01:
        return None, "Refund amount does not match the calculated amount."

    return order, None


# ---------------------------------------------------------
# Tool 4: Issue refund
# ---------------------------------------------------------

def issue_refund(order_id, refund_inr):
    """Record an approved simulated refund in JSONL format."""

    amount = round(float(refund_inr), 2)

    order, error = validate_refund_action(order_id, amount)

    if error:
        return {"error": error}

    if ledger_entry_exists(order_id, "issue_refund"):
        return {
            "order_id": order_id,
            "refund_inr": amount,
            "message": "Refund was already recorded.",
        }

    record = {
        "timestamp": timestamp_now(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": amount,
        "status": "simulated_refund_issued",
    }

    append_jsonl(LEDGER_PATH, record)

    return {
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Simulated refund successfully recorded.",
    }


# ---------------------------------------------------------
# Tool 5: Escalate to human
# ---------------------------------------------------------

def escalate_to_human(order_id, refund_inr):
    """Record an eligible high-value refund for human review."""

    amount = round(float(refund_inr), 2)

    order = get_order(order_id)

    if order.get("error"):
        return {"error": order["error"]}

    if order["status"] != "completed":
        return {"error": "Only completed orders can be escalated."}

    decision = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"],
    )

    if decision["decision"] != "human_review":
        return {
            "error": "Order does not meet the human escalation rule."
        }

    if abs(amount - decision["refund_inr"]) > 0.01:
        return {"error": "Escalation amount does not match the order."}

    if ledger_entry_exists(order_id, "escalate_to_human"):
        return {
            "order_id": order_id,
            "refund_inr": amount,
            "message": "Escalation was already recorded.",
        }

    record = {
        "timestamp": timestamp_now(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": amount,
        "status": "human_review_required",
    }

    append_jsonl(LEDGER_PATH, record)

    return {
        "order_id": order_id,
        "refund_inr": amount,
        "message": "Order successfully escalated for human review.",
    }


# ---------------------------------------------------------
# Tool specifications
# ---------------------------------------------------------

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order in the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant Instamart policy chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the eligible refund amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"},
                },
                "required": [
                    "order_amount",
                    "delivery_minutes",
                ],
            },
        },
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
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a high-value refund for human review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------------------------------------------------------
# Tool-call helpers
# ---------------------------------------------------------

def get_tool_call_details(tool_call):
    """Support the tool-call object format returned by mock_llm."""

    name = getattr(tool_call, "name", None)
    arguments = getattr(tool_call, "arguments", None)
    call_id = getattr(tool_call, "id", None)

    if name is None and isinstance(tool_call, dict):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        call_id = tool_call.get("id")

    # Also support a nested function-style representation.
    function = getattr(tool_call, "function", None)

    if function is not None:
        name = getattr(function, "name", name)
        arguments = getattr(function, "arguments", arguments)

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        arguments = {}

    return str(name), arguments, str(call_id or "")


def execute_tool(name, arguments, last_order):
    if name not in TOOL_MAP:
        return {"error": f"Unknown tool: {name}"}

    if name == "calculate_refund":
        if not last_order or last_order.get("error"):
            return {"error": "A verified order is required first."}

        if last_order["status"] != "completed":
            return {"error": "Only completed orders can be refunded."}

        return calculate_refund(
            arguments["order_amount"],
            arguments["delivery_minutes"],
        )

    if name in ("issue_refund", "escalate_to_human"):
        if not last_order or last_order.get("error"):
            return {"error": "A verified order is required first."}

        if arguments.get("order_id") != last_order.get("order_id"):
            return {"error": "Order ID does not match the verified order."}

    return TOOL_MAP[name](**arguments)


# ---------------------------------------------------------
# ReAct messages and checkpoints
# ---------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are an Instamart refund AGENT. "
    "Use the available tools to verify orders, retrieve policy, "
    "calculate refunds, issue eligible refunds, or escalate high-value "
    "refunds. Only completed orders are eligible. A delivery must exceed "
    "45 minutes for a refund. Refunds above INR 1,000 must be escalated. "
    "Do not invent order information. If an order is missing, return "
    "NOT_FOUND. If an order is ineligible, return DENIED. "
    "If the process cannot safely finish, return a FALLBACK status. "
    "Your final response must begin with one of these statuses: "
    "REFUND, ESCALATE, DENIED, NOT_FOUND, FALLBACK_MAX_ITERATIONS, "
    "FALLBACK_TOKEN_LIMIT, or FALLBACK_TIMEOUT."
)


class SimulatedCrash(Exception):
    pass


def save_checkpoint(state):
    path = checkpoint_path(state["run_id"])
    write_json(path, state)


def make_initial_state(
    user_message,
    scenario_id,
    run_id,
    max_iterations,
    token_limit,
    time_limit_seconds,
):
    return {
        "run_id": safe_run_id(run_id),
        "scenario_id": str(scenario_id),
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": str(user_message),
            },
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "last_order": None,
        "max_iterations": int(max_iterations),
        "token_limit": int(token_limit),
        "time_limit_seconds": float(time_limit_seconds),
    }


def fallback_result(state, status, explanation):
    final_answer = f"{status}: {explanation}"

    return {
        "scenario_id": state["scenario_id"],
        "run_id": state["run_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
    }


def build_result(state, final_answer):
    status = str(final_answer).split(":", 1)[0].strip()

    return {
        "scenario_id": state["scenario_id"],
        "run_id": state["run_id"],
        "status": status,
        "final_answer": str(final_answer),
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
    }


# ---------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------

def continue_agent(state, crash_after_tool_count=None):
    started_at = time.monotonic()

    max_iterations = state["max_iterations"]
    token_limit = state["token_limit"]
    time_limit = state["time_limit_seconds"]

    while state["llm_calls"] < max_iterations:

        # Time guard: check before every model call.
        elapsed = time.monotonic() - started_at

        if elapsed >= time_limit:
            return fallback_result(
                state,
                "FALLBACK_TIMEOUT",
                "The agent reached its time limit. Human review is required.",
            )

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC,
        )

        state["llm_calls"] += 1

        call_tokens = int(response.usage.total_tokens)
        state["tokens_used"] += call_tokens

        # Token guard: stop before executing any requested tool.
        if state["tokens_used"] > token_limit:
            return fallback_result(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "The token budget was exceeded. Human review is required.",
            )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # No tool calls means the model has produced its final response.
        if not tool_calls:
            final_answer = message.content or (
                "FALLBACK_TIMEOUT: The model returned an empty response."
            )

            return build_result(state, final_answer)

        assistant_tool_calls = []
        parsed_calls = []

        for tool_call in tool_calls:
            name, arguments, call_id = get_tool_call_details(tool_call)

            assistant_tool_calls.append({
                "id": call_id,
                "name": name,
                "arguments": arguments,
            })

            parsed_calls.append((name, arguments, call_id))

        # Append the assistant message before the tool observations.
        state["messages"].append({
            "role": "assistant",
            "tool_calls": assistant_tool_calls,
        })

        for name, arguments, call_id in parsed_calls:

            try:
                observation = execute_tool(
                    name,
                    arguments,
                    state["last_order"],
                )

            except Exception as exc:
                observation = {
                    "error": f"Tool execution failed: {type(exc).__name__}: {exc}"
                }

            if name == "get_order":
                state["last_order"] = observation

            state["step_number"] += 1
            state["tools_executed"].append(name)

            step = {
                "step_number": state["step_number"],
                "tool": name,
                "arguments": arguments,
                "observation": observation,
            }

            state["steps"].append(step)

            # Append the tool observation to the conversation.
            state["messages"].append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(
                    observation,
                    ensure_ascii=False,
                    default=str,
                ),
            })

            # Save after every tool observation.
            save_checkpoint(state)

            append_jsonl(
                TRACE_PATH,
                {
                    "timestamp": timestamp_now(),
                    "run_id": state["run_id"],
                    "scenario_id": state["scenario_id"],
                    **step,
                },
            )

            # Simulated crash happens only after checkpoint is saved.
            if (
                crash_after_tool_count is not None
                and state["step_number"] >= crash_after_tool_count
            ):
                raise SimulatedCrash(
                    f"Simulated crash after tool observation "
                    f"{state['step_number']}"
                )

    return fallback_result(
        state,
        "FALLBACK_MAX_ITERATIONS",
        "The agent reached the maximum number of iterations. "
        "Human review is required.",
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_count=None,
):
    """
    Start a new ReAct run.

    A checkpoint is written after every tool observation.
    """

    ensure_paths()

    if run_id is None:
        run_id = f"run_{scenario_id}"

    state = make_initial_state(
        user_message=user_message,
        scenario_id=scenario_id,
        run_id=run_id,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )

    save_checkpoint(state)

    return continue_agent(
        state,
        crash_after_tool_count=crash_after_tool_count,
    )


def resume_agent(run_id):
    """Load a checkpoint and continue without repeating completed tools."""

    ensure_paths()

    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for run ID: {run_id}"
        )

    state = read_json(path)

    return continue_agent(state)


# ---------------------------------------------------------
# Scenario loading and result saving
# ---------------------------------------------------------

def load_scenarios():
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(SCENARIOS_PATH)

    frame = pd.read_csv(
        SCENARIOS_PATH,
        keep_default_na=False,
    )

    required = {
        "scenario_id",
        "customer_id",
        "message",
        "expected_outcome",
        "expected_refund_inr",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Scenario file is missing columns: {sorted(missing)}"
        )

    return frame.to_dict(orient="records")


def save_agent_runs(results):
    with AGENT_RUNS_PATH.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False, default=str) + "\n"
            )


def save_results_csv(results):
    rows = []

    for result in results:
        order_id = ""

        for step in result["steps"]:
            if step["tool"] == "get_order":
                order_id = step["observation"].get("order_id", "")
                break

        refund_amount = 0.0

        for step in result["steps"]:
            if step["tool"] in (
                "issue_refund",
                "escalate_to_human",
            ):
                refund_amount = float(
                    step["observation"].get("refund_inr", 0) or 0
                )

        rows.append({
            "scenario_id": result["scenario_id"],
            "run_id": result["run_id"],
            "status": result["status"],
            "order_id": order_id,
            "agent_response": result["final_answer"],
            "refund_inr": refund_amount,
            "tool_steps": len(result["steps"]),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
        })

    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)


def save_ledger_csv_compatibility():
    """Keep the earlier CSV output available for project compatibility."""

    records = read_ledger_records()

    pd.DataFrame(
        records,
        columns=[
            "timestamp",
            "tool",
            "order_id",
            "refund_inr",
            "status",
        ],
    ).to_csv(
        OUTPUT_DIR / "simulated_refund_ledger.csv",
        index=False,
    )


# ---------------------------------------------------------
# Safety demonstrations
# ---------------------------------------------------------

def run_safety_demos(s1_message):
    demos = []

    # Token budget demonstration.
    token_result = run_agent(
        user_message=s1_message,
        scenario_id="S1_TOKEN_LIMIT_DEMO",
        run_id="S1_token_limit_demo",
        max_iterations=6,
        token_limit=200,
        time_limit_seconds=10,
    )

    demos.append({
        "demo": "S1 token limit 200",
        **token_result,
    })

    # Immediate timeout demonstration.
    timeout_result = run_agent(
        user_message=s1_message,
        scenario_id="S1_TIMEOUT_DEMO",
        run_id="S1_timeout_demo",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=0,
    )

    demos.append({
        "demo": "S1 time limit 0 seconds",
        **timeout_result,
    })

    with SAFETY_DEMOS_PATH.open("w", encoding="utf-8") as file:
        for demo in demos:
            file.write(
                json.dumps(demo, ensure_ascii=False, default=str) + "\n"
            )

    return demos


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ensure_paths()

    scenarios = load_scenarios()

    if len(scenarios) != 7:
        raise ValueError(
            f"Expected seven agent scenarios; found {len(scenarios)}"
        )

    # Reset mock counters once for this independent batch.
    mock_llm.reset()

    # Start a clean demonstration ledger.
    LEDGER_PATH.write_text("", encoding="utf-8")
    TRACE_PATH.write_text("", encoding="utf-8")

    results = []

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        message = str(scenario["message"])

        print(f"\nRunning {scenario_id}...")

        if scenario_id == "S1":
            run_id = "S1_crash_resume_demo"

            try:
                run_agent(
                    user_message=message,
                    scenario_id=scenario_id,
                    run_id=run_id,
                    max_iterations=6,
                    token_limit=6000,
                    time_limit_seconds=10,
                    crash_after_tool_count=2,
                )

            except SimulatedCrash as exc:
                print(exc)
                print("Loading saved checkpoint and resuming S1...")

            result = resume_agent(run_id)

        else:
            result = run_agent(
                user_message=message,
                scenario_id=scenario_id,
                run_id=f"scenario_{scenario_id}",
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
            )

        results.append(result)

        print("Status:", result["status"])
        print("Model calls:", result["llm_calls"])
        print("Tokens used:", result["tokens_used"])
        print("Tools:", result["tools_executed"])

    # Save the seven primary scenario runs.
    save_agent_runs(results)
    save_results_csv(results)
    save_ledger_csv_compatibility()

    # Run the two independent safety demonstrations.
    s1_message = str(scenarios[0]["message"])
    demos = run_safety_demos(s1_message)

    print("\n--- Safety demonstrations ---")

    for demo in demos:
        print(
            demo["demo"],
            "=>",
            demo["status"],
            "| model calls:",
            demo["llm_calls"],
            "| tokens:",
            demo["tokens_used"],
        )

    print("\n--- Final scenario results ---")

    for result in results:
        print(
            result["scenario_id"],
            result["status"],
            "calls:",
            result["llm_calls"],
            "tokens:",
            result["tokens_used"],
        )

    print("\nSaved files:")
    print("outputs/agent_runs.jsonl")
    print("outputs/refund_ledger.jsonl")
    print("outputs/refund_agent_results.csv")
    print("outputs/simulated_refund_ledger.csv")
    print("outputs/refund_agent_trace.jsonl")
    print("outputs/agent_safety_demos.jsonl")
    print("state/  (per-run checkpoints)")


if __name__ == "__main__":
    main()
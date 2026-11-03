from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part_d_refund_agent import (
    SCENARIOS_PATH,
    RESULTS_PATH,
    OUTPUT_DIR,
    get_order,
    search_policy,
    calculate_refund,
    MAX_ITERATIONS,
    load_scenarios,
)

COMPARISON_PATH = OUTPUT_DIR / "agent_architecture_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "agent_architecture_summary.csv"


def plan_and_execute(scenario):
    """Read-only Plan-and-Execute simulation."""

    scenario_id = str(scenario["scenario_id"])
    message = str(scenario["message"])

    result = {
        "scenario_id": scenario_id,
        "plan_execute_status": "",
        "plan_execute_refund_inr": 0.0,
        "planning_steps": 1,
        "execution_tool_calls": 0,
        "finalization_steps": 1,
        "plan_execute_total_steps": 0,
        "planned_actions": "",
        "explanation": "",
    }

    # Repeated-action scenario.
    if "[LOOP]" in message:
        result.update({
            "plan_execute_status": "FALLBACK_MAX_ITERATIONS",
            "execution_tool_calls": MAX_ITERATIONS - 2,
            "planned_actions": "Detect repeated action; stop safely",
            "explanation": (
                "The planned execution was stopped by the "
                "maximum-iteration guard."
            ),
        })

        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )

        return result

    # Find order ID.
    import re

    match = re.search(r"ORD-\d{6}", message)

    if not match:
        result.update({
            "plan_execute_status": "NOT_FOUND",
            "planned_actions": "Request a valid order ID",
            "explanation": "No valid order ID was present.",
        })
        result["plan_execute_total_steps"] = 2
        return result

    order_id = match.group(0)

    # Step 1: Look up order.
    result["execution_tool_calls"] += 1
    order = get_order(order_id)

    if order.get("error"):
        result.update({
            "plan_execute_status": "NOT_FOUND",
            "planned_actions": "Look up order; stop if missing",
            "explanation": "The order was not found in SQLite.",
        })

        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )

        return result

    # Step 2: Retrieve policy.
    result["execution_tool_calls"] += 1
    policy = search_policy("refund policy late delivery")

    if not policy.get("results"):
        result.update({
            "plan_execute_status": "FALLBACK_POLICY",
            "planned_actions": "Look up order; search policy; stop",
            "explanation": "No relevant policy chunks were retrieved.",
        })

        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )

        return result

    # Only completed orders are eligible.
    if order["status"] != "completed":
        result.update({
            "plan_execute_status": "DENIED",
            "planned_actions": "Look up order; check policy; deny",
            "explanation": "Only completed orders are eligible.",
        })

        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )

        return result

    # Step 3: Calculate refund.
    result["execution_tool_calls"] += 1

    decision = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"],
    )

    result["plan_execute_refund_inr"] = float(
        decision["refund_inr"]
    )

    if decision["decision"] == "denied":
        result.update({
            "plan_execute_status": "DENIED",
            "planned_actions": (
                "Look up order; retrieve policy; calculate; deny"
            ),
            "explanation": decision["reason"],
        })

    elif decision["decision"] == "human_review":
        result.update({
            "plan_execute_status": "ESCALATE",
            "planned_actions": (
                "Look up order; retrieve policy; calculate; escalate"
            ),
            "explanation": decision["reason"],
        })

    elif decision["decision"] == "eligible":
        result["execution_tool_calls"] += 1

        result.update({
            "plan_execute_status": "REFUND",
            "planned_actions": (
                "Look up order; retrieve policy; calculate; "
                "validate; simulate refund"
            ),
            "explanation": (
                "Refund qualifies under the demo rules. "
                "No ledger entry is written by this comparison."
            ),
        })

    else:
        result.update({
            "plan_execute_status": "FALLBACK_POLICY",
            "planned_actions": "Stop and request human review",
            "explanation": "Unrecognized decision; fail safely.",
        })

    result["plan_execute_total_steps"] = (
        result["planning_steps"]
        + result["execution_tool_calls"]
        + result["finalization_steps"]
    )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "ReAct results not found. Run part_d_refund_agent.py first."
        )

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(SCENARIOS_PATH)

    react = pd.read_csv(RESULTS_PATH)
    react["scenario_id"] = react["scenario_id"].astype(str)

    react_lookup = react.set_index("scenario_id").to_dict(
        orient="index"
    )

    scenarios = load_scenarios()
    rows = []

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])

        if scenario_id not in react_lookup:
            raise ValueError(
                f"Missing ReAct result for {scenario_id}"
            )

        react_result = react_lookup[scenario_id]
        plan_result = plan_and_execute(scenario)

        react_status = str(react_result["status"])
        plan_status = str(plan_result["plan_execute_status"])

        rows.append({
            "scenario_id": scenario_id,
            "react_status": react_status,
            "plan_execute_status": plan_status,
            "outcome_match": react_status == plan_status,
            "react_refund_inr": float(
                react_result.get("refund_inr", 0) or 0
            ),
            "plan_execute_refund_inr": float(
                plan_result["plan_execute_refund_inr"]
            ),
            "react_model_calls": int(
                react_result.get("llm_calls", 0) or 0
            ),
            "react_tokens_used": int(
                react_result.get("tokens_used", 0) or 0
            ),
            "react_tool_steps": int(
                react_result.get("tool_steps", 0) or 0
            ),
            "plan_execute_planning_steps": int(
                plan_result["planning_steps"]
            ),
            "plan_execute_tool_calls": int(
                plan_result["execution_tool_calls"]
            ),
            "plan_execute_total_steps": int(
                plan_result["plan_execute_total_steps"]
            ),
            "planned_actions": plan_result["planned_actions"],
            "explanation": plan_result["explanation"],
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_PATH, index=False)

    summary = pd.DataFrame([
        {
            "metric": "Scenarios compared",
            "react": len(comparison),
            "plan_execute": len(comparison),
        },
        {
            "metric": "Matching outcome statuses",
            "react": int(comparison["outcome_match"].sum()),
            "plan_execute": int(comparison["outcome_match"].sum()),
        },
        {
            "metric": "Outcome match rate (%)",
            "react": round(
                100 * comparison["outcome_match"].mean(), 2
            ),
            "plan_execute": round(
                100 * comparison["outcome_match"].mean(), 2
            ),
        },
        {
            "metric": "Total ReAct model calls",
            "react": int(comparison["react_model_calls"].sum()),
            "plan_execute": None,
        },
        {
            "metric": "Total ReAct tokens",
            "react": int(comparison["react_tokens_used"].sum()),
            "plan_execute": None,
        },
        {
            "metric": "Average Plan-and-Execute steps",
            "react": None,
            "plan_execute": round(
                comparison["plan_execute_total_steps"].mean(), 2
            ),
        },
        {
            "metric": "Automatic refund outcomes",
            "react": int((comparison["react_status"] == "REFUND").sum()),
            "plan_execute": int(
                (comparison["plan_execute_status"] == "REFUND").sum()
            ),
        },
        {
            "metric": "Human escalation outcomes",
            "react": int(
                (comparison["react_status"] == "ESCALATE").sum()
            ),
            "plan_execute": int(
                (comparison["plan_execute_status"] == "ESCALATE").sum()
            ),
        },
        {
            "metric": "Maximum-iteration fallbacks",
            "react": int(
                (
                    comparison["react_status"]
                    == "FALLBACK_MAX_ITERATIONS"
                ).sum()
            ),
            "plan_execute": int(
                (
                    comparison["plan_execute_status"]
                    == "FALLBACK_MAX_ITERATIONS"
                ).sum()
            ),
        },
    ])

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n--- ReAct vs Plan-and-Execute ---")

    print(
        comparison[
            [
                "scenario_id",
                "react_status",
                "plan_execute_status",
                "outcome_match",
                "react_model_calls",
                "react_tokens_used",
                "plan_execute_total_steps",
            ]
        ].to_string(index=False)
    )

    print("\n--- Summary ---")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(COMPARISON_PATH)
    print(SUMMARY_PATH)

    print(
        "\nNote: Plan-and-Execute is a deterministic, read-only "
        "simulation. Its execution steps are not equivalent to "
        "ReAct model calls or token usage."
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
import re

import pandas as pd

# ---------------------------------------------------------
# Paths and project imports
# ---------------------------------------------------------

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
    MAX_STEPS,
)

COMPARISON_PATH = OUTPUT_DIR / "agent_architecture_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "agent_architecture_summary.csv"


# ---------------------------------------------------------
# Plan-and-Execute simulation
# ---------------------------------------------------------

def plan_and_execute(scenario):
    """
    Read-only Plan-and-Execute simulation.

    The planner creates a sequence of steps first.
    The executor then checks the order and policy, calculates
    eligibility, and determines the safe next action.

    This function does not write to the refund ledger.
    """

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

    # Simulate a repeated-action request and apply the step guard.
    if "[LOOP]" in message:
        result.update({
            "plan_execute_status": "step_limit",
            "planning_steps": 1,
            "execution_tool_calls": MAX_STEPS - 2,
            "finalization_steps": 1,
            "planned_actions": "Detect repeated action; stop safely",
            "explanation": (
                "Repeated-action scenario stopped by the maximum-step guard."
            ),
        })

        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )
        return result

    match = re.search(r"ORD-\d{6}", message)

    if not match:
        result.update({
            "plan_execute_status": "needs_info",
            "planned_actions": "Request order ID",
            "explanation": "No valid order ID was present in the request.",
        })
        result["plan_execute_total_steps"] = 2
        return result

    order_id = match.group(0)

    # Execute step 1: verify the order.
    result["execution_tool_calls"] += 1
    order = get_order(order_id)

    if order.get("error"):
        result.update({
            "plan_execute_status": "order_not_found",
            "planned_actions": "Look up order; stop if missing",
            "explanation": "The order could not be found in SQLite.",
        })
        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )
        return result

    # Execute step 2: retrieve relevant policy information.
    result["execution_tool_calls"] += 1
    policy = search_policy("refund policy late delivery")

    if not policy.get("found"):
        result.update({
            "plan_execute_status": "human_review",
            "planned_actions": "Look up order; search policy; escalate",
            "explanation": "No matching policy text was found.",
        })
        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )
        return result

    # Only completed orders are eligible for this demo.
    if order["status"] != "completed":
        result.update({
            "plan_execute_status": "denied",
            "planned_actions": "Look up order; check policy; deny",
            "explanation": "Only completed orders can be refunded.",
        })
        result["plan_execute_total_steps"] = (
            result["planning_steps"]
            + result["execution_tool_calls"]
            + result["finalization_steps"]
        )
        return result

    # Execute step 3: calculate eligibility from verified data.
    result["execution_tool_calls"] += 1

    decision = calculate_refund(
        order.get("order_amount"),
        order.get("delivery_minutes"),
    )

    result["plan_execute_refund_inr"] = float(
        decision.get("refund_inr", 0.0)
    )

    if decision["decision"] == "denied":
        result.update({
            "plan_execute_status": "denied",
            "planned_actions": (
                "Look up order; check policy; calculate; deny"
            ),
            "explanation": decision["reason"],
        })

    elif decision["decision"] == "human_review":
        result.update({
            "plan_execute_status": "human_review",
            "planned_actions": (
                "Look up order; check policy; calculate; escalate"
            ),
            "explanation": decision["reason"],
        })

    elif decision["decision"] == "eligible":
        # Simulate the action, without writing to the ledger.
        result["execution_tool_calls"] += 1
        result.update({
            "plan_execute_status": "refund_response",
            "planned_actions": (
                "Look up order; check policy; calculate; "
                "validate refund; simulate issue"
            ),
            "explanation": (
                "Refund meets the demo eligibility rules. "
                "No ledger entry was written by this comparison."
            ),
        })

    else:
        result.update({
            "plan_execute_status": "human_review",
            "planned_actions": "Stop and request human review",
            "explanation": "Unrecognized decision; fail safely.",
        })

    result["plan_execute_total_steps"] = (
        result["planning_steps"]
        + result["execution_tool_calls"]
        + result["finalization_steps"]
    )

    return result


# ---------------------------------------------------------
# Compare both approaches on the same scenarios
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"ReAct results not found: {RESULTS_PATH}\n"
            "Run src/part_d_refund_agent.py first."
        )

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIOS_PATH}"
        )

    react = pd.read_csv(RESULTS_PATH).fillna({
        "order_id": "",
        "agent_response": "",
        "refund_inr": 0,
        "tool_steps": 0,
    })

    # Use the scenario loader from the existing agent module.
    from part_d_refund_agent import load_scenarios

    scenarios = load_scenarios()

    react["scenario_id"] = react["scenario_id"].astype(str)

    react_lookup = react.set_index("scenario_id").to_dict(
        orient="index"
    )

    rows = []

    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        react_result = react_lookup.get(scenario_id)

        if react_result is None:
            raise ValueError(
                f"Missing ReAct result for {scenario_id}. "
                "Rerun the refund agent before comparing."
            )

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
            "react_loop_turns": int(
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
            "metric": "Average ReAct loop turns",
            "react": round(
                comparison["react_loop_turns"].mean(), 2
            ),
            "plan_execute": None,
        },
        {
            "metric": "Average Plan-and-Execute total steps",
            "react": None,
            "plan_execute": round(
                comparison["plan_execute_total_steps"].mean(), 2
            ),
        },
        {
            "metric": "Automatic refund outcomes",
            "react": int(
                (comparison["react_status"] == "refund_response").sum()
            ),
            "plan_execute": int(
                (
                    comparison["plan_execute_status"]
                    == "refund_response"
                ).sum()
            ),
        },
        {
            "metric": "Human-review outcomes",
            "react": int(
                (comparison["react_status"] == "human_review").sum()
            ),
            "plan_execute": int(
                (
                    comparison["plan_execute_status"]
                    == "human_review"
                ).sum()
            ),
        },
        {
            "metric": "Safe loop stops",
            "react": int(
                (comparison["react_status"] == "step_limit").sum()
            ),
            "plan_execute": int(
                (comparison["plan_execute_status"] == "step_limit").sum()
            ),
        },
    ])

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n--- Scenario-by-scenario comparison ---")
    print(
        comparison[
            [
                "scenario_id",
                "react_status",
                "plan_execute_status",
                "outcome_match",
                "react_loop_turns",
                "plan_execute_total_steps",
            ]
        ].to_string(index=False)
    )

    print("\n--- Summary ---")
    print(summary.to_string(index=False))

    print(f"\nComparison saved to: {COMPARISON_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(
        "\nNote: Plan-and-Execute is a deterministic, read-only "
        "simulation. Its step counts are not directly equivalent "
        "to ReAct model turns or token usage."
    )


if __name__ == "__main__":
    main()

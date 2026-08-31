import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents import (
    run_flu_planner_agent,
    run_flu_analyst_agent,
)

from tools import (
    summarize_flu_dataset,
    build_flu_summary_for_scope,
)


EVALUATION_TASKS = [
    {
        "name": "cross_season",
        "expected_scope": "cross_season",
        "task": """
Compare influenza hospitalization patterns across seasons.

Describe how national peak timing changes from season to season
and whether jurisdiction-level peak timing differs across years.
""",
    },
    {
        "name": "latest_season",
        "expected_scope": "latest_season",
        "task": """
Focus only on the most recent influenza season.

Summarize the latest season's hospitalization pattern and describe
which jurisdictions peaked earlier or later than the dominant peak.
""",
    },
    {
        "name": "all_seasons",
        "expected_scope": "all_seasons",
        "task": """
Perform a detailed analysis of every available influenza season.

For each season, examine national peak behavior, jurisdiction-level
peak timing, and peak-rate variation.
""",
    },
]


REQUIRED_ANALYST_KEYS = {
    "supported_findings",
    "seasonality_summary",
    "spatiotemporal_summary",
    "cross_season_changes",
    "notable_geographic_patterns",
    "limitations",
    "hypotheses_requiring_external_data",
    "needs_more_detail",
    "requested_season",
    "reason",
}


def evaluate_scope(
    expected_scope,
    selected_scope,
):
    return selected_scope == expected_scope


def evaluate_required_keys(analysis):
    missing_keys = REQUIRED_ANALYST_KEYS - set(
        analysis.keys()
    )

    return {
        "passed": len(missing_keys) == 0,
        "missing_keys": sorted(missing_keys),
    }


def evaluate_requested_season(
    analysis,
    selected_scope,
    valid_seasons,
    latest_season,
):
    needs_more_detail = analysis.get(
        "needs_more_detail"
    )

    requested_season = analysis.get(
        "requested_season"
    )

    if not needs_more_detail:
        return {
            "passed": requested_season is None,
            "reason": (
                "No additional detail requested."
            ),
        }

    if requested_season not in valid_seasons:
        return {
            "passed": False,
            "reason": (
                f"Invalid requested season: "
                f"{requested_season}"
            ),
        }

    if (
        selected_scope == "latest_season"
        and requested_season != latest_season
    ):
        return {
            "passed": False,
            "reason": (
                "Latest-season scope requested detail "
                f"for {requested_season} instead of "
                f"{latest_season}."
            ),
        }

    return {
        "passed": True,
        "reason": (
            f"Requested season {requested_season} "
            "is valid for the selected scope."
        ),
    }


def evaluate_output_lengths(analysis):
    checks = {
        "supported_findings": 6,
        "notable_geographic_patterns": 4,
        "limitations": 4,
        "hypotheses_requiring_external_data": 3,
    }

    failures = []

    for key, maximum in checks.items():
        value = analysis.get(key, [])

        if not isinstance(value, list):
            failures.append(
                f"{key} is not a list"
            )
            continue

        if len(value) > maximum:
            failures.append(
                f"{key} has {len(value)} items "
                f"(maximum {maximum})"
            )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
    }


def run_evaluation():
    dataset_summary = summarize_flu_dataset()

    valid_seasons = dataset_summary["seasons"]
    latest_season = valid_seasons[-1]

    evaluation_results = []

    for test in EVALUATION_TASKS:
        print("\n" + "=" * 70)
        print(f"EVALUATING: {test['name']}")
        print("=" * 70)

        result = {
            "test_name": test["name"],
            "expected_scope": test[
                "expected_scope"
            ],
            "selected_scope": None,
            "planner_scope_pass": False,
            "required_keys_pass": False,
            "feedback_scope_pass": False,
            "output_length_pass": False,
            "workflow_pass": False,
            "details": {},
        }

        try:
            planning_request = (
                run_flu_planner_agent(
                    test["task"],
                    dataset_summary,
                )
            )

            selected_scope = planning_request[
                "analysis_scope"
            ]

            result["selected_scope"] = (
                selected_scope
            )

            scope_pass = evaluate_scope(
                test["expected_scope"],
                selected_scope,
            )

            result["planner_scope_pass"] = (
                scope_pass
            )

            flu_summary = (
                build_flu_summary_for_scope(
                    selected_scope
                )
            )

            analysis = run_flu_analyst_agent(
                test["task"],
                planning_request,
                flu_summary,
            )

            required_key_result = (
                evaluate_required_keys(
                    analysis
                )
            )

            feedback_result = (
                evaluate_requested_season(
                    analysis,
                    selected_scope,
                    valid_seasons,
                    latest_season,
                )
            )

            output_length_result = (
                evaluate_output_lengths(
                    analysis
                )
            )

            result["required_keys_pass"] = (
                required_key_result["passed"]
            )

            result["feedback_scope_pass"] = (
                feedback_result["passed"]
            )

            result["output_length_pass"] = (
                output_length_result["passed"]
            )

            result["workflow_pass"] = True

            result["details"] = {
                "needs_more_detail": analysis.get(
                    "needs_more_detail"
                ),
                "requested_season": analysis.get(
                    "requested_season"
                ),
                "required_keys": (
                    required_key_result
                ),
                "feedback_scope": (
                    feedback_result
                ),
                "output_lengths": (
                    output_length_result
                ),
            }

        except Exception as error:
            result["workflow_pass"] = False
            result["details"]["error"] = str(error)

        evaluation_results.append(result)

        print(json.dumps(result, indent=2))

    return evaluation_results


def print_summary(results):
    total = len(results)

    planner_passes = sum(
        result["planner_scope_pass"]
        for result in results
    )

    feedback_passes = sum(
        result["feedback_scope_pass"]
        for result in results
    )

    required_key_passes = sum(
        result["required_keys_pass"]
        for result in results
    )

    length_passes = sum(
        result["output_length_pass"]
        for result in results
    )

    workflow_passes = sum(
        result["workflow_pass"]
        for result in results
    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Planner scope accuracy:      "
        f"{planner_passes}/{total}"
    )

    print(
        f"Feedback scope compliance:   "
        f"{feedback_passes}/{total}"
    )

    print(
        f"Required output keys:        "
        f"{required_key_passes}/{total}"
    )

    print(
        f"Output length compliance:    "
        f"{length_passes}/{total}"
    )

    print(
        f"Workflow completion:         "
        f"{workflow_passes}/{total}"
    )


def save_evaluation(results):
    output_dir = Path("logs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_path = (
        output_dir
        / f"evaluation_{timestamp}.json"
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print("\n--- EVALUATION SAVED ---")
    print(f"Saved to: {output_path}")


def main():
    results = run_evaluation()

    print_summary(results)

    save_evaluation(results)


if __name__ == "__main__":
    main()
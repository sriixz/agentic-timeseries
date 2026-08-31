import json

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


TEST_TASKS = [
    {
        "name": "cross_season",
        "task": """
Compare influenza hospitalization patterns across seasons.

Describe how national peak timing changes from season to season
and whether jurisdiction-level peak timing differs across years.
""",
    },
    {
        "name": "latest_season",
        "task": """
Focus only on the most recent influenza season.

Summarize the latest season's hospitalization pattern and describe
which jurisdictions peaked earlier or later than the dominant peak.
""",
    },
    {
        "name": "all_seasons",
        "task": """
Perform a detailed analysis of every available influenza season.

For each season, examine national peak behavior, jurisdiction-level
peak timing, and peak-rate variation.
""",
    },
]


def main():
    dataset_summary = summarize_flu_dataset()

    results = []

    for test in TEST_TASKS:
        print("\n" + "=" * 70)
        print(f"TEST: {test['name']}")
        print("=" * 70)

        planning_request = run_flu_planner_agent(
            test["task"],
            dataset_summary,
        )

        selected_scope = planning_request[
            "analysis_scope"
        ]

        print("\n--- PLANNER DECISION ---")
        print(
            json.dumps(
                planning_request,
                indent=2,
            )
        )

        flu_summary = build_flu_summary_for_scope(
            selected_scope
        )

        print("\n--- SUMMARY TYPE BUILT ---")
        print(f"analysis_scope: {selected_scope}")
        print(
            "top-level keys:",
            list(flu_summary.keys()),
        )

        analysis = run_flu_analyst_agent(
            test["task"],
            planning_request,
            flu_summary,
        )

        print("\n--- ANALYST RESULT ---")
        print(
            json.dumps(
                analysis,
                indent=2,
            )
        )

        results.append(
            {
                "test_name": test["name"],
                "expected_intent": test["name"],
                "selected_scope": selected_scope,
                "planner_reason": planning_request[
                    "reason"
                ],
                "needs_more_detail": analysis[
                    "needs_more_detail"
                ],
                "requested_season": analysis[
                    "requested_season"
                ],
            }
        )

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
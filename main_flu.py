import json
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from agents import (
    run_flu_planner_agent,
    run_flu_analyst_agent,
    run_flu_second_pass_analyst,
)

from tools import (
    summarize_flu_dataset,
    build_flu_summary_for_scope,
    build_detailed_season_summary,
)

from logger import save_log


def main():
    user_task = """
Analyze the CDC FluSight influenza hospitalization dataset.

Summarize the major seasonal patterns in influenza hospitalizations.

Determine whether there are meaningful spatiotemporal patterns,
including differences in when influenza activity peaks across
jurisdictions and how those patterns change across seasons.
"""

    run_log = {
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "user_task": user_task.strip(),
        "models": {
            "planner": "gpt-5.4-mini",
            "analyst": "claude-sonnet-4-5",
        },
        "dataset_summary": None,
        "planner": None,
        "selected_scope": None,
        "selected_summary": None,
        "first_analysis": None,
        "feedback": None,
        "season_detail": None,
        "final_analysis": None,
        "error": None,
    }

    try:
        # ----------------------------------------------------------
        # STEP 1: Python builds only dataset metadata
        # ----------------------------------------------------------

        print("\n--- BUILDING DATASET METADATA ---")

        dataset_summary = summarize_flu_dataset()

        run_log["dataset_summary"] = dataset_summary

        print(
            json.dumps(
                dataset_summary,
                indent=2,
            )
        )

        # ----------------------------------------------------------
        # STEP 2: GPT decides the analytical scope
        # ----------------------------------------------------------

        planning_request = run_flu_planner_agent(
            user_task,
            dataset_summary,
        )

        run_log["planner"] = planning_request

        print("\n--- FLUSIGHT PLANNER AGENT ---")
        print(
            json.dumps(
                planning_request,
                indent=2,
            )
        )

        analysis_scope = planning_request[
            "analysis_scope"
        ]

        run_log["selected_scope"] = analysis_scope

        # ----------------------------------------------------------
        # STEP 3: Python builds data for GPT-selected scope
        # ----------------------------------------------------------

        print("\n--- BUILDING PLANNER-SELECTED SUMMARY ---")
        print(f"Selected scope: {analysis_scope}")

        flu_summary = build_flu_summary_for_scope(
            analysis_scope
        )

        run_log["selected_summary"] = flu_summary

        print(
            json.dumps(
                flu_summary,
                indent=2,
            )
        )

        # ----------------------------------------------------------
        # STEP 4: Claude performs first-pass analysis
        # ----------------------------------------------------------

        analysis = run_flu_analyst_agent(
            user_task,
            planning_request,
            flu_summary,
        )

        run_log["first_analysis"] = analysis

        print("\n--- FIRST FLUSIGHT ANALYSIS ---")
        print(
            json.dumps(
                analysis,
                indent=2,
            )
        )

        # ----------------------------------------------------------
        # STEP 5: Optional feedback
        # ----------------------------------------------------------

        if analysis["needs_more_detail"]:
            requested_season = analysis[
                "requested_season"
            ]

            valid_seasons = dataset_summary[
                "seasons"
            ]

            if requested_season not in valid_seasons:
                raise ValueError(
                    f"Analyst requested invalid season: "
                    f"{requested_season}"
                )

            print("\n--- FEEDBACK TRIGGERED ---")
            print(
                f"Analyst requested deeper detail for: "
                f"{requested_season}"
            )

            run_log["feedback"] = {
                "triggered": True,
                "requested_season": requested_season,
                "reason": analysis["reason"],
            }

            # ------------------------------------------------------
            # STEP 6: Python builds detailed requested-season data
            # ------------------------------------------------------

            season_detail = (
                build_detailed_season_summary(
                    requested_season
                )
            )

            run_log["season_detail"] = {
                "season": season_detail["season"],
                "jurisdiction_count": (
                    season_detail[
                        "jurisdiction_count"
                    ]
                ),
                "dominant_peak_date": (
                    season_detail[
                        "peak_timing"
                    ]["dominant_peak_date"]
                ),
                "dominant_peak_count": (
                    season_detail[
                        "peak_timing"
                    ]["dominant_peak_count"]
                ),
                "peak_date_distribution": (
                    season_detail[
                        "peak_timing"
                    ]["distribution"]
                ),
                "early_count": (
                    season_detail[
                        "timing_groups"
                    ]["early_count"]
                ),
                "typical_count": (
                    season_detail[
                        "timing_groups"
                    ]["typical_count"]
                ),
                "late_count": (
                    season_detail[
                        "timing_groups"
                    ]["late_count"]
                ),
                "peak_rate_statistics": (
                    season_detail[
                        "peak_rate_statistics"
                    ]
                ),
                "timing_group_rate_statistics": (
                    season_detail[
                        "timing_group_rate_statistics"
                    ]
                ),
                "peak_rate_extremes": (
                    season_detail[
                        "peak_rate_extremes"
                    ]
                ),
            }

            print("\n--- DETAILED SEASON RETRIEVAL ---")
            print(
                json.dumps(
                    run_log["season_detail"],
                    indent=2,
                )
            )

            # ------------------------------------------------------
            # STEP 7: Claude performs second-pass analysis
            # ------------------------------------------------------

            final_analysis = (
                run_flu_second_pass_analyst(
                    user_task,
                    analysis,
                    season_detail,
                )
            )

        else:
            run_log["feedback"] = {
                "triggered": False,
            }

            final_analysis = analysis

        # ----------------------------------------------------------
        # STEP 8: Final result
        # ----------------------------------------------------------

        run_log["final_analysis"] = final_analysis
        run_log["status"] = "success"

        print("\n--- FINAL FLUSIGHT ANALYSIS ---")
        print(
            json.dumps(
                final_analysis,
                indent=2,
            )
        )

    except Exception as error:
        run_log["status"] = "failed"
        run_log["error"] = str(error)

        print("\n--- FLUSIGHT WORKFLOW ERROR ---")
        print(str(error))

    finally:
        save_log(run_log)


if __name__ == "__main__":
    main()
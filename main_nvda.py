import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from agents import (
    run_retriever_agent,
    run_analyst_agent,
    run_second_pass_analyst,
)
from tools import fetch_stock_data
from logger import save_log


def main():
    user_task = """
Analyze NVIDIA's recent stock-price behavior.
Determine whether the recent movement looks unusual
and whether more historical context would be useful.
"""

    run_log = {
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "user_task": user_task.strip(),
        "models": {
            "retriever": "gpt-5.4-mini",
            "analyst": "claude-sonnet-4-5",
        },
        "retriever": None,
        "first_retrieval": None,
        "first_analysis": None,
        "feedback": None,
        "second_retrieval": None,
        "final_analysis": None,
        "error": None,
    }

    try:
        # Step 1: Retriever Agent decides what data is needed
        retrieval_request = run_retriever_agent(user_task)
        run_log["retriever"] = retrieval_request

        print("\n--- RETRIEVER AGENT ---")
        print(json.dumps(retrieval_request, indent=2))

        # Step 2: Retrieve the requested financial data
        data = fetch_stock_data(
            retrieval_request["symbol"],
            retrieval_request["period"],
        )

        run_log["first_retrieval"] = {
            "symbol": data["symbol"],
            "period": data["period"],
            "observation_count": len(data["observations"]),
        }

        print("\n--- FIRST DATA RETRIEVAL ---")
        print(json.dumps(run_log["first_retrieval"], indent=2))

        # Step 3: Claude analyzes the initial dataset
        analysis = run_analyst_agent(
            user_task,
            retrieval_request,
            data,
        )

        run_log["first_analysis"] = analysis

        print("\n--- FIRST ANALYSIS ---")
        print(json.dumps(analysis, indent=2))

        # Step 4: Feedback loop
        if analysis["needs_more_data"]:
            requested_period = analysis["requested_period"]

            print("\n--- FEEDBACK TRIGGERED ---")
            print(
                f"Analyst requested more historical data: "
                f"{requested_period}"
            )

            run_log["feedback"] = {
                "triggered": True,
                "requested_period": requested_period,
                "reason": analysis["reason"],
            }

            # Retrieve expanded dataset
            expanded_data = fetch_stock_data(
                retrieval_request["symbol"],
                requested_period,
            )

            run_log["second_retrieval"] = {
                "symbol": expanded_data["symbol"],
                "period": expanded_data["period"],
                "observation_count": len(
                    expanded_data["observations"]
                ),
            }

            print("\n--- SECOND DATA RETRIEVAL ---")
            print(json.dumps(run_log["second_retrieval"], indent=2))

            # Second-pass analysis
            final_analysis = run_second_pass_analyst(
                user_task,
                analysis,
                expanded_data,
            )

        else:
            run_log["feedback"] = {
                "triggered": False,
            }

            final_analysis = analysis

        # Step 5: Final result
        run_log["final_analysis"] = final_analysis
        run_log["status"] = "success"

        print("\n--- FINAL ANALYSIS ---")
        print(json.dumps(final_analysis, indent=2))

    except Exception as error:
        run_log["status"] = "failed"
        run_log["error"] = str(error)

        print("\n--- WORKFLOW ERROR ---")
        print(str(error))

    finally:
        save_log(run_log)


if __name__ == "__main__":
    main()
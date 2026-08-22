import json
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from agents import (
    run_retriever_agent,
    run_analyst_agent,
    run_second_pass_analyst
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
            "analyst": "claude-sonnet-4-5"
        },
        "retriever": None,
        "first_retrieval": None,
        "first_analysis": None,
        "feedback": None,
        "second_retrieval": None,
        "final_analysis": None,
        "error": None
    }

    try:
        # -------------------------
        # 1. RETRIEVER AGENT
        # -------------------------

        retrieval_request = run_retriever_agent(user_task)
        run_log["retriever"] = retrieval_request

        print("\n--- RETRIEVER AGENT ---")
        print(json.dumps(retrieval_request, indent=2))

        # -------------------------
        # 2. DATA TOOL
        # -------------------------

        data = fetch_stock_data(
            retrieval_request["symbol"],
            retrieval_request["period"]
        )

        run_log["first_retrieval"] = {
            "symbol": data["symbol"],
            "period": data["period"],
            "observation_count": len(data["observations"])
        }

        print("\n--- DATA TOOL ---")
        print(
            f"Retrieved {len(data['observations'])} observations "
            f"for {data['symbol']} over {data['period']}."
        )

        # -------------------------
        # 3. ANALYST AGENT
        # -------------------------

        analysis = run_analyst_agent(
            user_task,
            retrieval_request,
            data
        )

        run_log["first_analysis"] = analysis

        print("\n--- ANALYST AGENT ---")
        print(json.dumps(analysis, indent=2))

        # -------------------------
        # 4. FEEDBACK LOOP
        # -------------------------

        if analysis["needs_more_data"]:

            print("\n--- FEEDBACK LOOP ---")
            print(
                f"Analyst requested more data: "
                f"{analysis['requested_period']}"
            )

            run_log["feedback"] = {
                "triggered": True,
                "requested_period": analysis["requested_period"],
                "reason": analysis["reason"]
            }

            expanded_data = fetch_stock_data(
                retrieval_request["symbol"],
                analysis["requested_period"]
            )

            run_log["second_retrieval"] = {
                "symbol": expanded_data["symbol"],
                "period": expanded_data["period"],
                "observation_count": len(expanded_data["observations"])
            }

            print(
                f"Retrieved {len(expanded_data['observations'])} observations "
                f"for second-pass analysis."
            )

            final_analysis = run_second_pass_analyst(
                user_task,
                analysis,
                expanded_data
            )

        else:
            run_log["feedback"] = {
                "triggered": False
            }

            final_analysis = analysis

        # -------------------------
        # 5. FINAL RESULT
        # -------------------------

        run_log["final_analysis"] = final_analysis
        run_log["status"] = "success"

        print("\n--- FINAL ANALYSIS ---")
        print(json.dumps(final_analysis, indent=2))

    except Exception as error:
        run_log["status"] = "failed"
        run_log["error"] = str(error)

        print("\n--- WORKFLOW ERROR ---")
        print(error)

    finally:
        # Save the run whether it succeeded or failed
        save_log(run_log)


if __name__ == "__main__":
    main()
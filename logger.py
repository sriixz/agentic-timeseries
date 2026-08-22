import json
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")


def save_log(run_log):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = LOG_DIR / f"run_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(run_log, file, indent=2)

    print("\n--- LOG SAVED ---")
    print(f"Saved execution log to: {filename}")
import json

from tools import build_flu_analysis_summary


def main():
    summary = build_flu_analysis_summary()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
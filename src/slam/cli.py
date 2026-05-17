import argparse
import importlib


def record() -> int:
    importlib.import_module("slam.pipelines.acquisition")
    return 0


def reconstruct() -> int:
    importlib.import_module("slam.pipelines.reconstruction")
    return 0


def regression() -> int:
    from slam.pipelines.benchmarking import main as harness_main

    return harness_main([])


def regression_score() -> int:
    from slam.pipelines.benchmarking import main as harness_main

    return harness_main(["--no-run"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SLAM project command runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("record", help="Run the acquisition pipeline")
    subparsers.add_parser("reconstruct", help="Run the reconstruction pipeline")
    subparsers.add_parser("regression", help="Run reconstruction benchmarks")
    subparsers.add_parser("regression-score", help="Score existing benchmark outputs")

    args = parser.parse_args(argv)
    commands = {
        "record": record,
        "reconstruct": reconstruct,
        "regression": regression,
        "regression-score": regression_score,
    }
    return commands[args.command]()

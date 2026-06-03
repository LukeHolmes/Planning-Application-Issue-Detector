from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from planning_detector.config import PipelineConfig, default_data_path
from planning_detector.predict import score_file
from planning_detector.train import METRICS_NAME, MODELS_DIR, train_pipeline


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train planning application issue detector")
    parser.add_argument("--data", type=Path, default=None, help="Path to Excel/CSV dataset")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--output", type=Path, default=MODELS_DIR, help="Model output directory")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config) if args.config else PipelineConfig.load()
    data = args.data or default_data_path()

    _, metrics, meta = train_pipeline(data, config, args.output)
    print(json.dumps({"metrics": metrics.to_dict(), "meta": meta}, indent=2))
    print(f"\nSaved model to {args.output / 'planning_detector.joblib'}")
    print(f"Metrics written to {args.output / METRICS_NAME}")


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Score planning applications")
    parser.add_argument("data", type=Path, help="Path to Excel/CSV to score")
    parser.add_argument("--output", type=Path, default=None, help="Write CSV results here")
    parser.add_argument("--artifact", type=Path, default=None, help="Trained model path")
    args = parser.parse_args()

    result = score_file(args.data, args.artifact)
    cols = [c for c in result.columns if c in ("predicted_problematic", "problematic_probability")]
    print(result[cols].head(20).to_string())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m planning_detector.cli [train|predict] ...")
        sys.exit(1)
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "train":
        train_main()
    elif cmd == "predict":
        predict_main()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

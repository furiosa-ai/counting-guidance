import argparse
from collections import defaultdict
from pathlib import Path
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", nargs="+")
    parser.add_argument("--no_cats", action="store_true")
    args = parser.parse_args()

    metrics = {"clip": defaultdict(lambda: defaultdict(lambda: {})), "blip": defaultdict(lambda: defaultdict(lambda: {}))}

    paths = []

    for path in args.path:
        if "*" in path:
            paths += list(Path().glob(path))
        else:
            paths.append(path)

    for path in paths:
        if args.no_cats:
            sub_paths = [Path(path)]
        else:
            sub_paths = [p for p in Path(path).glob("*/") if p.is_dir()]

        for sub_path in sub_paths:
            cat = sub_path.name

            with open(sub_path / "clip_aggregated_metrics.json", "r") as f:
                clip_metric = json.load(f)["full_text_aggregation"]

            with open(sub_path / "blip_aggregated_metrics.json", "r") as f:
                blip_metric = json.load(f)["average_similarity"]
            # blip_metric = None

            metrics["clip"][cat][path] = clip_metric
            metrics["blip"][cat][path] = blip_metric

            print(sub_path, clip_metric, blip_metric)

    pd.DataFrame(metrics["clip"]).to_csv("metrics_clip.csv")
    pd.DataFrame(metrics["blip"]).to_csv("metrics_blip.csv")


if __name__ == "__main__":
    main()

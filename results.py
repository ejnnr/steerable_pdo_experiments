import argparse
from pathlib import Path

import numpy as np
import yaml

# this script takes in a folder path and then recursively collects all
# results.yaml files in that directory. It averages them and prints
# summary statistics

parser = argparse.ArgumentParser(description="Analyze the results")
parser.add_argument("path", type=str, help="path to the folder containing the results")

args = parser.parse_args()

results = []
keys = set()

for path in Path(args.path).rglob("results.yaml"):
    with open(path, "r") as file:
        results.append(yaml.safe_load(file))
        keys = keys.union(results[-1].keys())

print(f"Found {len(results)} files with {len(keys)} different metrics\n")

output = {}
for key in keys:
    vals = [result[key] for result in results if key in result]
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals))

    output[key] = {
        "N runs": n,
        "mean": round(mean, 3),
        "std": round(std, 3)
    }

print(yaml.dump(output))
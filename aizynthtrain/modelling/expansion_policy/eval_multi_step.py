"""Module routines for evaluating one-step retrosynthesis model"""
import argparse
import tempfile
import json
import subprocess
from collections import defaultdict
from typing import Sequence, Optional, Dict, Any

import yaml
import pandas as pd
import numpy as np
from route_distances.ted.reactiontree import ReactionTreeWrapper

from aizynthtrain.utils.configs import (
    ExpansionModelEvaluationConfig,
    load_config,
)


def _create_config(
    model_path: str, templates_path: str, stock_path: str, properties: Dict[str, Any]
) -> str:
    _, filename = tempfile.mkstemp(suffix=".yaml")
    dict_ = {
        "expansion": {"default": [model_path, templates_path]},
        "stock": {"default": stock_path},
        "search": properties,
    }
    with open(filename, "w") as fileobj:
        yaml.dump(dict_, fileobj)
    return filename


def _eval_finding(
    config: ExpansionModelEvaluationConfig, finder_config_path: str
) -> None:
    output_path = config.filename("finder_output").replace(".hdf5", "_finding.hdf5")
    _run_finder(config.target_smiles, finder_config_path, output_path)

    finder_output = pd.read_hdf(output_path, "table")
    stats = {
        "target": [str(x) for x in finder_output["target"].to_list()],
        "first solution time": [
            float(x) for x in finder_output["first_solution_time"].to_list()
        ],
        "is solved": [bool(x) for x in finder_output["is_solved"].to_list()],
    }

    return stats


def _eval_recovery(
    config: ExpansionModelEvaluationConfig, finder_config_path: str
) -> None:
    with open(config.reference_routes, "r") as fileobj:
        ref_trees = json.load(fileobj)
    smiles = [tree["smiles"] for tree in ref_trees]

    _, smiles_filename = tempfile.mkstemp(suffix=".txt")
    with open(smiles_filename, "w") as fileobj:
        fileobj.write("\n".join(smiles))

    output_path = config.filename("finder_output").replace(".hdf5", "_recovery.hdf5")
    _run_finder(smiles_filename, finder_config_path, output_path)

    finder_output = pd.read_hdf(output_path, "table")
    stats = defaultdict(list)
    for ref_tree, (_, row) in zip(ref_trees, finder_output.iterrows()):
        ref_wrapper = ReactionTreeWrapper(ref_tree)
        dists = [
            ReactionTreeWrapper(dict_).distance_to(ref_wrapper) for dict_ in row.trees
        ]
        min_dists = float(min(dists))
        stats["target"].append(ref_tree["smiles"])
        stats["is solved"].append(bool(row.is_solved))
        stats["found reference"].append(min_dists == 0.0)
        stats["closest to reference"].append(min_dists)
        stats["rank of closest"].append(float(np.argmin(dists)))

    return stats


def _run_finder(
    smiles_filename: str, finder_config_path: str, output_path: str
) -> None:
    subprocess.run(
        [
            "aizynthcli",
            "--smiles",
            smiles_filename,
            "--config",
            finder_config_path,
            "--output",
            output_path,
        ]
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface to multi-step evaluation"""
    parser = argparse.ArgumentParser(
        "Tool evaluate a multi-step retrosynthesis model using AiZynthFinder"
    )
    parser.add_argument(
        "--model_path", help="overrides the model path from the config file"
    )
    parser.add_argument(
        "--templates_path", help="overrides the templates path from the config file"
    )
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: ExpansionModelEvaluationConfig = load_config(
        args.config, "expansion_model_evaluation"
    )

    all_stats = {}

    if config.stock_for_finding and config.target_smiles:
        finder_config_path = _create_config(
            args.model_path or config.filename("onnx_model"),
            args.templates_path or config.filename("unique_templates"),
            config.stock_for_finding,
            config.search_properties_for_finding,
        )

        stats = _eval_finding(config, finder_config_path)
        all_stats["finding"] = stats

    if config.stock_for_recovery and config.reference_routes:
        finder_config_path = _create_config(
            args.model_path or config.filename("onnx_model"),
            args.templates_path or config.filename("unique_templates"),
            config.stock_for_recovery,
            config.search_properties_for_recovery,
        )

        stats = _eval_recovery(config, finder_config_path)
        all_stats["recovery"] = stats

    with open(config.filename("multistep_report"), "w") as fileobj:
        json.dump(all_stats, fileobj)

    if "finding" in all_stats:
        print("\nEvaluation of route finding capabilities:")
        pd_stats = pd.DataFrame(all_stats["finding"])
        print(
            f"Average first solution time: {pd_stats['first solution time'].mean():.2f}"
        )
        print(
            f"Average number of solved target: {pd_stats['is solved'].mean()*100:.2f}%"
        )

    if "recovery" in all_stats:
        print("\nEvaluation of route recovery capabilities:")
        pd_stats = pd.DataFrame(all_stats["recovery"])
        print(
            f"Average number of solved target: {pd_stats['is solved'].mean()*100:.2f}%"
        )
        print(f"Average found reference: {pd_stats['found reference'].mean()*100:.2f}%")
        print(
            f"Average closest to reference: {pd_stats['closest to reference'].mean():.2f}"
        )
        print(f"Average rank of closest: {pd_stats['rank of closest'].mean():.2f}")


if __name__ == "__main__":
    main()

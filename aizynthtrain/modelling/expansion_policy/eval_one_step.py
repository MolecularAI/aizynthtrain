""" Module routines for evaluating one-step retrosynthesis model
"""
import argparse
import math
import random
import tempfile
import json
from collections import defaultdict
from typing import Sequence, Optional, Dict, List, Any

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from aizynthfinder.aizynthfinder import AiZynthExpander

from aizynthtrain.utils.configs import (
    ExpansionModelEvaluationConfig,
    load_config,
)

YAML_TEMPLATE = """expansion:
  default:
      - {}
      - {}
"""


def _create_config(model_path: str, templates_path: str) -> str:
    _, filename = tempfile.mkstemp(suffix=".yaml")
    with open(filename, "w") as fileobj:
        fileobj.write(YAML_TEMPLATE.format(model_path, templates_path))
    return filename


def _create_test_reaction(config: ExpansionModelEvaluationConfig) -> str:
    """
    Selected reactions for testing. Group reactions by parent reaction classification
    and then selected random reactions from each group.
    """

    def transform_rsmi(row):
        reactants, _, products = row[config.columns.reaction_smiles].split(">")
        rxn = AllChem.ReactionFromSmarts(f"{reactants}>>{products}")
        AllChem.RemoveMappingNumbersFromReactions(rxn)
        return AllChem.ReactionToSmiles(rxn)

    test_lib_path = config.filename("library", "testing")
    data = pd.read_csv(test_lib_path, sep="\t")
    if config.columns.ring_breaker not in data.columns:
        data[config.columns.ring_breaker] = [False] * len(data)
    trunc_class = data[config.columns.classification].apply(
        lambda x: ".".join(x.split(" ")[0].split(".")[:2])
    )

    class_to_idx = defaultdict(list)
    for idx, val in enumerate(trunc_class):
        class_to_idx[val].append(idx)
    n_per_class = math.ceil(config.n_test_reactions / len(class_to_idx))

    random.seed(1789)
    selected_idx = []
    for indices in class_to_idx.values():
        if len(indices) > n_per_class:
            selected_idx.extend(random.sample(indices, k=n_per_class))
        else:
            selected_idx.extend(indices)

    data_sel = data.iloc[selected_idx]
    rsmi = data_sel.apply(transform_rsmi, axis=1)
    filename = test_lib_path.replace(".csv", "_selected.csv")
    pd.DataFrame(
        {
            config.columns.reaction_smiles: rsmi,
            config.columns.ring_breaker: data_sel[config.columns.ring_breaker],
            "original_index": selected_idx,
        }
    ).to_csv(filename, index=False, sep="\t")
    return filename


def _eval_expander(
    expander_output: List[Dict[str, Any]],
    ref_reactions_path: str,
    config: ExpansionModelEvaluationConfig,
) -> None:
    ref_reactions = pd.read_csv(ref_reactions_path, sep="\t")

    stats = defaultdict(list)
    for (_, row), output in zip(ref_reactions.iterrows(), expander_output):
        reactants, _, product = row[config.columns.reaction_smiles].split(">")
        nrings_prod = CalcNumRings(Chem.MolFromSmiles(product))
        reactants_inchis = set(
            Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in reactants.split(".")
        )
        found = False
        found_idx = None
        ring_broken = False
        for idx, outcome in enumerate(output["outcome"]):
            outcome_inchis = set(
                Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
                for smi in outcome.split(".")
            )
            if outcome_inchis == reactants_inchis:
                found = True
                found_idx = idx + 1
            nrings_reactants = sum(
                CalcNumRings(Chem.MolFromSmiles(smi)) for smi in outcome.split(".")
            )
            if nrings_reactants < nrings_prod:
                ring_broken = True
        stats["target"].append(product)
        stats["expected reactants"].append(reactants)
        stats["found expected"].append(found)
        stats["rank of expected"].append(found_idx)
        if row[config.columns.ring_breaker]:
            stats["ring broken"].append(ring_broken)
        else:
            stats["ring broken"].append(None)
        stats["ring breaking"].append(bool(row[config.columns.ring_breaker]))
        stats["non-applicable"].append(output["non-applicable"])

    with open(config.filename("onestep_report"), "w") as fileobj:
        json.dump(stats, fileobj)

    stats = pd.DataFrame(stats)
    print(f"Evaluated {len(ref_reactions)} reactions")
    print(f"Average found expected: {stats['found expected'].mean()*100:.2f}%")
    print(f"Average rank of expected: {stats['rank of expected'].mean():.2f}")
    print(f"Average ring broken when expected: {stats['ring broken'].mean()*100:.2f}%")
    print(f"Percentage of ring reactions: {stats['ring breaking'].mean()*100:.2f}%")
    print(f"Average non-applicable (in top-50): {stats['non-applicable'].mean():.2f}")


def _run_expander(
    ref_reactions_path: str, config_path: str, config: ExpansionModelEvaluationConfig
) -> List[Dict[str, Any]]:
    ref_reactions = pd.read_csv(ref_reactions_path, sep="\t")
    targets = [
        rxn.split(">")[-1] for rxn in ref_reactions[config.columns.reaction_smiles]
    ]

    expander = AiZynthExpander(configfile=config_path)
    expander.expansion_policy.select("default")
    expander_output = []
    for target in tqdm(targets):
        outcome = expander.do_expansion(target, config.top_n)
        outcome_list = [
            item.reaction_smiles().split(">>")[1]
            for item_list in outcome
            for item in item_list
        ]
        expander_output.append(
            {
                "outcome": outcome_list,
                "non-applicable": expander.stats["non-applicable"],
            }
        )

    with open(config.filename("expander_output"), "w") as fileobj:
        json.dump(expander_output, fileobj, indent=4)

    return expander_output


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface to multi-step evaluation"""
    parser = argparse.ArgumentParser(
        "Tool to evaluate a one-step retrosynthesis model using AiZynthExpander"
    )
    parser.add_argument(
        "--model_path", help="overrides the model path from the config file"
    )
    parser.add_argument(
        "--templates_path", help="overrides the templates path from the config file"
    )
    parser.add_argument(
        "--test_library", help="overrides the test_library from the config file"
    )
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: ExpansionModelEvaluationConfig = load_config(
        args.config, "expansion_model_evaluation"
    )

    ref_reactions_path = args.test_library or _create_test_reaction(config)

    config_path = _create_config(
        args.model_path or config.filename("onnx_model"),
        args.templates_path or config.filename("unique_templates"),
    )

    expander_output = _run_expander(ref_reactions_path, config_path, config)

    _eval_expander(expander_output, ref_reactions_path, config)


if __name__ == "__main__":
    main()

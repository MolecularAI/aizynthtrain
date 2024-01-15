"""Module for splitting a Chemformer dataset into train / validation / test sets"""
import argparse
import pandas as pd

from aizynthtrain.utils.data_utils import split_data, extract_route_reactions
from aizynthtrain.utils.configs import (
    load_config,
    ChemformerDataPrepConfig,
)
from typing import Sequence, Optional


def split_dataset(config: ChemformerDataPrepConfig, dataset: pd.DataFrame) -> None:
    if config.routes_to_exclude:
        reaction_hashes = extract_route_reactions(config.routes_to_exclude)
        print(
            f"Found {len(reaction_hashes)} unique reactions given routes. Will make these test set",
            flush=True,
        )
        is_external = dataset[config.reaction_hash_col].isin(reaction_hashes)
        subdata = dataset[~is_external]
    else:
        is_external = None
        subdata = dataset
    train_indices = []
    val_indices = []
    test_indices = []

    subdata.apply(
        split_data,
        train_frac=config.training_fraction,
        random_seed=config.random_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    subdata[config.set_col] = "train"
    subdata.loc[val_indices, config.set_col] = "val"
    subdata.loc[test_indices, config.set_col] = "test"

    if is_external is None:
        subdata.to_csv(config.chemformer_data_path, sep="\t", index=False)
        return

    dataset.loc[~is_external, config.set_col] = subdata.set.values
    dataset.loc[is_external, config.set_col] = "test"

    dataset[config.is_external_col] = False
    dataset.loc[is_external, config.is_external_col] = True
    dataset.to_csv(config.chemformer_data_path, sep="\t", index=False)


def main(args: Optional[Sequence[str]]):
    """Command-line interface to the routines"""
    parser = argparse.ArgumentParser("Create dataset split for Chemformer data.")
    parser.add_argument("--config_path", default="The path to the configuration file")
    args = parser.parse_args(args=args)
    config: ChemformerDataPrepConfig = load_config(
        args.config_path, "chemformer_data_prep"
    )

    dataset = pd.read_csv(config.reaction_components_path, sep="\t")
    split_dataset(config, dataset)
    return


if __name__ == "__main__":
    main()

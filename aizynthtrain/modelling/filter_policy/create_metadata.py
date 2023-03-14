""" 
This module contains scripts to generate metadata artifacts for the filter policy

* A NPZ file with indices of the training, validation and testing partitions

"""
import argparse
from typing import Sequence, Optional

import pandas as pd
import numpy as np

from aizynthtrain.utils.configs import (
    FilterModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.data_utils import split_data


def _save_split_indices(
    dataset: pd.DataFrame, config: FilterModelPipelineConfig
) -> None:
    """Perform a split and save the indices to disc"""
    print("Creating split of dataset...", flush=True)

    test_indices = []
    train_indices = []
    val_indices = []
    split_data(
        dataset,
        train_frac=config.training_fraction,
        random_seed=config.random_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    print(
        f"Selecting {len(train_indices)} ({len(train_indices)/len(dataset)*100:.2f}%) records as training set"
    )
    print(
        f"Selecting {len(val_indices)} ({len(val_indices)/len(dataset)*100:.2f}%) records as validation set"
    )
    print(
        f"Selecting {len(test_indices)} ({len(test_indices)/len(dataset)*100:.2f}%) records as test set",
        flush=True,
    )

    np.savez(
        config.filename("split_indices"),
        train=train_indices,
        val=val_indices,
        test=test_indices,
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface of the routines"""
    parser = argparse.ArgumentParser("Create filter library metadata")
    parser.add_argument("config", default="The path to the configuration file")
    args = parser.parse_args(args=args)

    config: FilterModelPipelineConfig = load_config(
        args.config, "filter_model_pipeline"
    )

    dataset = pd.read_csv(
        config.filename("library"),
        sep="\t",
    )

    _save_split_indices(dataset, config)


if __name__ == "__main__":
    main()

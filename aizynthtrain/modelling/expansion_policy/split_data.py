"""Module routines for splitting data for expansion model"""
import argparse
from typing import Sequence, Optional, Union

import pandas as pd
import numpy as np
from scipy import sparse


from aizynthtrain.utils.configs import (
    ExpansionModelPipelineConfig,
    load_config,
)


def _split_and_save_data(
    data: Union[pd.DataFrame, np.ndarray, sparse.csr_matrix],
    data_label: str,
    config: ExpansionModelPipelineConfig,
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    test_arr: np.ndarray,
) -> None:
    array_dict = {"training": train_arr, "validation": val_arr, "testing": test_arr}
    for subset, arr in array_dict.items():
        filename = config.filename(data_label, subset=subset)
        if isinstance(data, pd.DataFrame):
            data.iloc[arr].to_csv(
                filename,
                sep="\t",
                index=False,
            )
        elif isinstance(data, np.ndarray):
            np.savez(filename, data[arr])
        else:
            sparse.save_npz(filename, data[arr], compressed=True)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface for the splitting routines"""
    parser = argparse.ArgumentParser(
        "Tool to split data to be used in training a expansion network policy"
    )
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: ExpansionModelPipelineConfig = load_config(
        args.config, "expansion_model_pipeline"
    )

    split_indices = np.load(config.filename("split_indices"))
    train_arr = split_indices["train"]
    val_arr = split_indices["val"]
    test_arr = split_indices["test"]

    print("Splitting template library...", flush=True)
    dataset = pd.read_csv(
        config.filename("library"),
        sep="\t",
    )
    _split_and_save_data(dataset, "library", config, train_arr, val_arr, test_arr)

    print("Splitting labels...", flush=True)
    data = sparse.load_npz(config.filename("model_labels"))
    _split_and_save_data(data, "model_labels", config, train_arr, val_arr, test_arr)

    print("Splitting inputs...", flush=True)
    data = sparse.load_npz(config.filename("model_inputs"))
    _split_and_save_data(data, "model_inputs", config, train_arr, val_arr, test_arr)


if __name__ == "__main__":
    main()

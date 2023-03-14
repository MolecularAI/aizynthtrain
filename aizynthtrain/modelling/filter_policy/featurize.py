"""Module that featurizes a library for training an filter model"""
import argparse
from typing import Sequence, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from aizynthtrain.utils.configs import (
    FilterModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.files import (
    read_csv_batch,
    save_numpy_array,
    save_sparse_matrix,
)
from aizynthtrain.utils.chem import smiles_to_fingerprint, reaction_to_fingerprint


def _make_inputs(
    config: FilterModelPipelineConfig,
    dataset: pd.DataFrame,
    batch: Tuple[int, int, int] = None,
) -> None:
    print("Generating inputs...", flush=True)
    smiles_column = config.library_columns.reaction_smiles

    products = np.asarray([smiles.split(">")[-1] for smiles in dataset[smiles_column]])
    reactants = np.asarray([smiles.split(">")[0] for smiles in dataset[smiles_column]])
    fp_kwargs = {
        "fp_length": config.model_hyperparams.fingerprint_length,
        "fp_radius": config.model_hyperparams.fingerprint_radius,
    }

    inputs = np.apply_along_axis(
        reaction_to_fingerprint, 0, [products, reactants], **fp_kwargs
    ).astype(np.int8)
    inputs = sparse.lil_matrix(inputs.T).tocsr()
    save_sparse_matrix(inputs, config.filename("model_inputs_rxn"), batch)

    inputs = np.apply_along_axis(
        smiles_to_fingerprint, 0, [products], **fp_kwargs
    ).astype(np.int8)
    inputs = sparse.lil_matrix(inputs.T).tocsr()
    save_sparse_matrix(inputs, config.filename("model_inputs_prod"), batch)


def _make_labels(
    config: FilterModelPipelineConfig,
    dataset: pd.DataFrame,
    batch: Tuple[int, int, int] = None,
) -> None:
    print("Generating labels...", flush=True)
    labels = dataset[config.library_columns.label].to_numpy()
    save_numpy_array(labels, config.filename("model_labels"), batch)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface for the featurization tool"""
    parser = argparse.ArgumentParser(
        "Tool to featurize a reaction library to be used in training a filter network policy"
    )
    parser.add_argument("config", default="The path to the configuration file")
    parser.add_argument("--batch", type=int, nargs=3, help="the batch specification")
    args = parser.parse_args(args)

    config: FilterModelPipelineConfig = load_config(
        args.config, "filter_model_pipeline"
    )

    dataset = read_csv_batch(
        config.filename("library"),
        batch=args.batch,
        sep="\t",
        usecols=[config.library_columns.reaction_smiles, config.library_columns.label],
    )

    _make_inputs(config, dataset, args.batch)
    _make_labels(config, dataset, args.batch)


if __name__ == "__main__":
    main()

"""Module that featurizes a template library for training an expansion model"""
import argparse
from typing import Sequence, Optional, Tuple

import numpy as np
from scipy import sparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from aizynthtrain.utils.configs import (
    ExpansionModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.files import read_csv_batch


def _smiles_to_fingerprint(
    args: Sequence[str], fp_radius: int, fp_length: int
) -> np.ndarray:
    mol = Chem.MolFromSmiles(args[0])
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_length)
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def _make_inputs(
    config: ExpansionModelPipelineConfig, batch: Tuple[int, int, int] = None
) -> None:
    print("Generating inputs...", flush=True)
    smiles_dataset = read_csv_batch(
        config.filename("library"),
        batch=batch,
        sep="\t",
        usecols=[config.library_config.columns.reaction_smiles],
    )
    products = np.asarray(
        [smiles.split(">")[-1] for smiles in smiles_dataset.squeeze("columns")]
    )

    inputs = np.apply_along_axis(
        _smiles_to_fingerprint,
        0,
        [products],
        fp_length=config.model_hyperparams.fingerprint_length,
        fp_radius=config.model_hyperparams.fingerprint_radius,
    )
    inputs = sparse.lil_matrix(inputs.T).tocsr()

    filename = config.filename("model_inputs")
    if batch is not None:
        filename = filename.replace(".npz", f".{batch[0]}.npz")
    sparse.save_npz(filename, inputs, compressed=True)


def _make_labels(
    config: ExpansionModelPipelineConfig, batch: Tuple[int, int, int] = None
) -> None:
    print("Generating labels...", flush=True)

    # Find out the maximum template code
    with open(config.filename("template_code"), "r") as fileobj:
        nlabels = max(int(code) for idx, code in enumerate(fileobj) if idx > 0) + 1

    template_code_data = read_csv_batch(config.filename("template_code"), batch=batch)
    template_codes = template_code_data.squeeze("columns").to_numpy()

    labels = sparse.lil_matrix((len(template_codes), nlabels), dtype=np.int8)
    labels[np.arange(len(template_codes)), template_codes] = 1
    labels = labels.tocsr()

    filename = config.filename("model_labels")
    if batch is not None:
        filename = filename.replace(".npz", f".{batch[0]}.npz")
    sparse.save_npz(filename, labels, compressed=True)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface for the featurization tool"""
    parser = argparse.ArgumentParser(
        "Tool to featurize a template library to be used in training a expansion network policy"
    )
    parser.add_argument("config", default="The path to the configuration file")
    parser.add_argument("--batch", type=int, nargs=3, help="the batch specification")
    args = parser.parse_args(args)

    config: ExpansionModelPipelineConfig = load_config(
        args.config, "expansion_model_pipeline"
    )

    _make_inputs(config, batch=args.batch)
    _make_labels(config, batch=args.batch)


if __name__ == "__main__":
    main()

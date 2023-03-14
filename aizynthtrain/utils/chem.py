from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_fingerprint(
    args: Sequence[str], fp_radius: int, fp_length: int
) -> np.ndarray:
    mol = Chem.MolFromSmiles(args[0])
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_length)
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def reaction_to_fingerprint(
    args: Sequence[str], fp_radius: int, fp_length: int
) -> np.ndarray:
    product_smiles, reactants_smiles = args
    product_fp = smiles_to_fingerprint([product_smiles], fp_radius, fp_length)

    reactants_fp_list = []
    for smiles in reactants_smiles.split("."):
        reactants_fp_list.append(smiles_to_fingerprint([smiles], fp_radius, fp_length))

    return (product_fp - sum(reactants_fp_list)).astype(np.int8)

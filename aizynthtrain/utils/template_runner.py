"""Module containing routines to extract templates from reactions"""
import argparse
from typing import Optional, Sequence

import pandas as pd
from rxnutils.chem.reaction import ChemicalReaction, ReactionException

from aizynthtrain.utils.files import read_csv_batch


def generate_templates(
    data: pd.DataFrame,
    radius: int,
    expand_ring: bool,
    expand_hetero: bool,
    ringbreaker_column: str,
    smiles_column: str,
) -> None:
    """
    Generate templates for the reaction in a given dataframe

    This function will add 3 columns to the dataframe
    * RetroTemplate: the extracted retro template
    * TemplateHash: a unique identifier based on fingerprint bits
    * TemplateError: if not None, will identicate a reason why the extraction failed

    :param data: the data with reactions
    :param radius: the radius to use, unless using Ringbreaker logic
    :param expand_ring: if True, will expand template with ring atoms
    :param expand_hetero: if True, will expand template with bonded heteroatoms
    :param ringbreaker_column: if given, will apply Rinbreaker logic to rows where this column is True
    :param smiles_column: the column with the atom-mapped reaction SMILES
    """

    def _row_apply(
        row: pd.Series,
        column: str,
        radius: int,
        expand_ring: bool,
        expand_hetero: bool,
        ringbreaker_column: str,
    ) -> pd.Series:
        rxn = ChemicalReaction(row[column], clean_smiles=False)
        general_error = {
            "RetroTemplate": None,
            "TemplateHash": None,
            "TemplateError": "General error",
        }

        if ringbreaker_column and row[ringbreaker_column]:
            expand_ring = True
            expand_hetero = True
            radius = 0
        elif ringbreaker_column and not row[ringbreaker_column]:
            expand_ring = False
            expand_hetero = False
        try:
            _, retro_template = rxn.generate_reaction_template(
                radius=radius, expand_ring=expand_ring, expand_hetero=expand_hetero
            )
        except ReactionException as err:
            general_error["TemplateError"] = str(err)
            return pd.Series(general_error)
        except Exception:
            general_error["TemplateError"] = "General error when generating template"
            return pd.Series(general_error)

        try:
            hash_ = retro_template.hash_from_bits()
        except Exception:
            general_error[
                "TemplateError"
            ] = "General error when generating template hash"
            return pd.Series(general_error)

        return pd.Series(
            {
                "RetroTemplate": retro_template.smarts,
                "TemplateHash": hash_,
                "TemplateError": None,
            }
        )

    template_data = data.apply(
        _row_apply,
        axis=1,
        radius=radius,
        expand_ring=expand_ring,
        expand_hetero=expand_hetero,
        ringbreaker_column=ringbreaker_column,
        column=smiles_column,
    )
    return data.assign(
        **{column: template_data[column] for column in template_data.columns}
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface for template extraction"""
    parser = argparse.ArgumentParser("Generate retrosynthesis templates")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--expand_ring", action="store_true", default=False)
    parser.add_argument("--expand_hetero", action="store_true", default=False)
    parser.add_argument("--ringbreaker_column", default="")
    parser.add_argument("--smiles_column", required=True)
    parser.add_argument("--batch", type=int, nargs=2)
    args = parser.parse_args(args=args)

    data = read_csv_batch(args.input_path, sep="\t", index_col=False, batch=args.batch)
    data = generate_templates(
        data,
        args.radius,
        args.expand_ring,
        args.expand_hetero,
        args.ringbreaker_column,
        args.smiles_column,
    )
    data.to_csv(args.output_path, index=False, sep="\t")


if __name__ == "__main__":
    main()

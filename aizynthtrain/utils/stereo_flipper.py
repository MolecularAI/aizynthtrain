"""Module containing routines to flip stereo centers for specific reactions"""
import argparse
import re
from typing import Optional, Sequence

import pandas as pd
from rxnutils.chem.template import ReactionTemplate
from rxnutils.data.batch_utils import read_csv_batch

# One or two @ characters in isolation
STEREOCENTER_REGEX = r"[^@]([@]{1,2})[^@]"


def _count_chiral_centres(row: pd.Series) -> int:
    prod_template = row.split(">>")[0]
    return len(re.findall(STEREOCENTER_REGEX, prod_template))


def _flip_chirality(row: pd.Series) -> pd.Series:
    """
    Change @@ to @ and vice versa in a retrosynthesis template
    and then create a new template and a hash for that template
    """
    dict_ = row.to_dict()
    prod_template = row["RetroTemplate"].split(">>")[0]
    nats = len(re.search(STEREOCENTER_REGEX, prod_template)[1])
    assert nats in [1, 2]
    if nats == 1:
        dict_["RetroTemplate"] = row["RetroTemplate"].replace("@", "@@")
    else:
        dict_["RetroTemplate"] = row["RetroTemplate"].replace("@@", "@")
    dict_["TemplateHash"] = ReactionTemplate(
        dict_["RetroTemplate"], direction="retro"
    ).hash_from_bits()
    return pd.Series(dict_)


def flip_stereo(data: pd.DataFrame, selection_query: str) -> pd.DataFrame:
    """
    Find templates with one stereo center and flip it thereby creating
    new templates. These templates are appended onto the existing dataframe.

    A column "FlippedStereo" will be added to indicate if a stereocenter
    was flipped.

    :param data: the template library
    :param selection_query: only flip the stereo for a subset of rows
    :returns: the concatenated dataframe.
    """
    sel_data = data.query(selection_query)
    sel_data = sel_data[~sel_data["RetroTemplate"].isna()]

    chiral_centers_count = sel_data["RetroTemplate"].apply(_count_chiral_centres)
    sel_flipping = chiral_centers_count == 1
    flipped_data = sel_data[sel_flipping].apply(_flip_chirality, axis=1)

    existing_hashes = set(sel_data["TemplateHash"])
    keep_flipped = flipped_data["TemplateHash"].apply(
        lambda hash_: hash_ not in existing_hashes
    )
    flipped_data = flipped_data[keep_flipped]

    all_data = pd.concat([data, flipped_data])
    flag_column = [False] * len(data) + [True] * len(flipped_data)
    return all_data.assign(FlippedStereo=flag_column)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface for stereo flipper"""
    parser = argparse.ArgumentParser("Generate flipped stereo centers")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--batch", type=int, nargs=2)
    args = parser.parse_args(args=args)

    data = read_csv_batch(args.input_path, sep="\t", index_col=False, batch=args.batch)
    data = flip_stereo(
        data,
        args.query,
    )
    data.to_csv(args.input_path, index=False, sep="\t")


if __name__ == "__main__":
    main()

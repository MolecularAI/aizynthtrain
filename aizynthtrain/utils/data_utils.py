"""Module containing utility routines for data manipulation"""
import json
import random
from typing import List, Set, Dict, Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_route_reactions(
    filenames: List[str], metadata_key="reaction_hash"
) -> Set[str]:
    """
    Extract reaction hashes from given routes

    :param filenames: the paths of the routes
    :param metadata_key: the key in the metadata with the reaction hash
    :return: the extracted hashes
    """

    def traverse_tree(tree_dict: Dict[str, Any], hashes: Set[str]) -> None:
        if tree_dict["type"] == "reaction":
            hashes.add(tree_dict["metadata"][metadata_key])
        for child in tree_dict.get("children", []):
            traverse_tree(child, hashes)

    reaction_hashes = set()
    for filename in filenames:
        with open(filename, "r") as fileobj:
            routes = json.load(fileobj)
        for route in routes:
            traverse_tree(route, reaction_hashes)

    return reaction_hashes


def split_data(
    group_data: pd.DataFrame,
    train_frac: float,
    random_seed: int,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: Optional[List[int]] = None,
) -> None:
    """
    Split a dataframe into training, validation and optionally test partitions

    The indices of the rows selected for the different partitions are added to the
    given lists.

    First split the data into a training and validation set with `train_frac` specifying
    the portion of the data to end up in the training partition.

    Second, if `test_indices` is provided take a random half of the validation as test
    partition. If only one row is in the validation partition after the first step, randomly
    assign it to test with a 50% probability.

    :params group_data: the data to split
    :params train_frac: the fractition of the data to go into the training partition
    :params random_seed: the seed of the random generator
    :params train_indices: the list of add the indices of the training partition
    :params val_indices: the list of add the indices of the validation partition
    :params test_indices: the list of add the indices of the testing partition
    """
    if len(group_data) > 1:
        train_arr, val_arr = train_test_split(
            group_data.index,
            train_size=train_frac,
            random_state=random_seed,
            shuffle=True,
        )
    else:
        train_arr = list(group_data.index)
        val_arr = []

    train_indices.extend(train_arr)
    if test_indices is None:
        val_indices.extend(val_arr)
        return

    if len(val_arr) > 1:
        val_arr, train_arr = train_test_split(
            val_arr, test_size=0.5, random_state=random_seed, shuffle=True
        )
        test_indices.extend(train_arr)
        val_indices.extend(val_arr)
    elif random.random() < 0.5:
        test_indices.extend(val_arr)
    else:
        val_indices.extend(val_arr)

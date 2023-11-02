import json

import pytest
import pandas as pd

from aizynthtrain.utils.data_utils import extract_route_reactions, split_data


@pytest.fixture
def save_trees(tmpdir):
    trees = [
        {
            "type": "mol",
            "children": [
                {"type": "reaction", "metadata": {"reaction_hash": "AAAA"}},
                {
                    "type": "reaction",
                    "metadata": {"reaction_hash": "BBBB"},
                    "children": [{"type": "mol"}, {"type": "mol"}],
                },
            ],
        },
        {
            "type": "mol",
            "children": [
                {
                    "type": "reaction",
                    "metadata": {"reaction_hash": "BBBB"},
                    "children": [
                        {"type": "mol"},
                        {
                            "type": "mol",
                            "children": [
                                {
                                    "type": "reaction",
                                    "metadata": {"reaction_hash": "CCC"},
                                }
                            ],
                        },
                    ],
                },
            ],
        },
    ]
    filenames = [str(tmpdir / f"tree{idx}.json") for idx in range(len(trees))]
    for tree, filename in zip(trees, filenames):
        with open(filename, "w") as fileobj:
            json.dump([tree], fileobj)
    return filenames


def test_extract_route_reactions(save_trees):
    filenames = save_trees

    assert extract_route_reactions(filenames[:1]) == {"AAAA", "BBBB"}
    assert extract_route_reactions(filenames[1:]) == {"CCC", "BBBB"}
    assert extract_route_reactions(filenames) == {"AAAA", "CCC", "BBBB"}


def test_split_even_in_two():
    df = pd.DataFrame({"A": [1, 2, 3, 4]})

    val_indices = []
    train_indices = []

    split_data(df, 0.5, 666, train_indices, val_indices)

    assert len(train_indices) == 2
    assert len(val_indices) == 2
    assert set(val_indices).intersection(train_indices) == set()


def test_split_even_in_three():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]})

    val_indices = []
    train_indices = []
    test_indices = []

    split_data(df, 0.34, 666, train_indices, val_indices, test_indices)

    assert len(train_indices) == 2
    assert len(val_indices) == 2
    assert len(test_indices) == 2
    assert set(val_indices).intersection(train_indices) == set()
    assert set(val_indices).intersection(test_indices) == set()


def test_split_uneven_three():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    val_indices = []
    train_indices = []
    test_indices = []

    split_data(df, 0.90, 666, train_indices, val_indices, test_indices)

    assert len(train_indices) == 9
    assert len(val_indices) + len(test_indices) == 1
    assert set(val_indices).intersection(train_indices) == set()
    assert set(val_indices).intersection(test_indices) == set()


def test_split_low_data():
    df = pd.DataFrame({"A": [1]})

    val_indices = []
    train_indices = []

    split_data(df, 0.99, 666, train_indices, val_indices)

    assert len(train_indices) == 1
    assert len(val_indices) == 0


"""Module containing various file utilities"""
import math
import os
from typing import List, Tuple, Any

import pandas as pd
from scipy import sparse


def _combine_batches(
    filename: str, nbatches: int, read_func: Any, write_func: Any, combine_func: Any
):
    data = None
    for idx in range(nbatches):
        temp_data, filename2 = read_func(filename, idx)
        if data is None:
            data = temp_data
        else:
            data = combine_func(data, temp_data)
        os.remove(filename2)
    write_func(data, filename)


def combine_csv_batches(filename: str, nbatches: int) -> None:
    """
    Combine CSV batches to one master file

    The batch files are removed from disc

    :param filename: the filename of the master file
    :param nbatches: the number of batches
    """

    def _read_csv(filename: str, idx: int) -> pd.DataFrame:
        filename2 = f"{filename}.{idx}"
        return pd.read_csv(filename2, sep="\t"), filename2

    def _write_csv(data: pd.DataFrame, filename: str) -> None:
        data.to_csv(filename, index=False, sep="\t")

    def _combine_csv(data: pd.DataFrame, temp_data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([data, temp_data])

    return _combine_batches(filename, nbatches, _read_csv, _write_csv, _combine_csv)


def combine_sparse_matrix_batches(filename: str, nbatches: int) -> None:
    """
    Combine sparse matrix batches to one master file

    The batch files are removed from disc

    :param filename: the filename of the master file
    :param nbatches: the number of batches
    """

    def _read_matrix(filename: str, idx: int) -> Any:
        filename2 = filename.replace(".npz", f".{idx}.npz")
        return sparse.load_npz(filename2), filename2

    def _write_matrix(data: Any, filename: str) -> None:
        sparse.save_npz(filename, data, compressed=True)

    def _combine_matrix(data: pd.DataFrame, temp_data: pd.DataFrame) -> pd.DataFrame:
        return sparse.vstack([data, temp_data])

    return _combine_batches(
        filename, nbatches, _read_matrix, _write_matrix, _combine_matrix
    )


def create_csv_batches(filename: str, nbatches: int) -> List[Tuple[int, int, int]]:
    """
    Create batches for reading a splitted CSV-file

    The batches will be in  the form of a tuple with three indices:
        * Batch index
        * Start index
        * End index

    :param filename: the CSV file to make batches of
    :param nbatches: the number of batches
    :return: the created batches
    """
    file_size = nlines(filename) - 1
    chunk_size = math.ceil(file_size / nbatches)
    return [(idx, idx * chunk_size, (idx + 1) * chunk_size) for idx in range(nbatches)]


def nlines(filename: str) -> int:
    """Count and return the number of lines in a file"""
    with open(filename, "rb") as fileobj:
        return sum(1 for line in fileobj)


def prefix_filename(prefix: str, postfix: str) -> str:
    """
    Construct pre- and post-fixed filename

    :param prefix: the prefix, can be empty
    :param postfix: the postfix
    :return: the concatenated string
    """
    if prefix:
        return prefix + "_" + postfix
    return postfix


def read_csv_batch(
    filename: str, batch: Tuple[int, ...] = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Read parts of a CSV file as specified by a batch

    :param filename: the path to the CSV file on disc
    :param batch: the batch specification as returned by `create_csv_batches`
    """
    if batch is None:
        return pd.read_csv(filename, **kwargs)
    if len(batch) == 3:
        _, batch_start, batch_end = batch
    elif len(batch) == 2:
        batch_start, batch_end = batch
    else:
        raise ValueError(f"The batch specification can only be 2 or 3 not {len(batch)}")
    return pd.read_csv(
        filename,
        nrows=batch_end - batch_start,
        skiprows=range(1, batch_start + 1),
        **kwargs,
    )

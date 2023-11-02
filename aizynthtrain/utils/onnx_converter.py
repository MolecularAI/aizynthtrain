"""Module to convert Keras models to ONNX."""

import argparse
from typing import Optional, Sequence

import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model

from aizynthtrain.utils import keras_utils


def convert_to_onnx(keras_model: str, onnx_model: str) -> None:
    """Converts a Keras model to ONNX.

    :param keras_model: the filename of the Keras model
    :param onnx_model: the filename to save the ONNX model at
    """
    custom_objects = {
        "top10_acc": keras_utils.top10_acc,
        "top50_acc": keras_utils.top50_acc,
        "tf": tf,
    }
    model = load_model(keras_model, custom_objects=custom_objects)

    tf2onnx.convert.from_keras(model, output_path=onnx_model)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line interface to convert a Keras model to Onnx."""
    parser = argparse.ArgumentParser("Tool to convert a Keras model to Onnx")
    parser.add_argument("keras_model", help="the Keras model file")
    parser.add_argument("onnx_model", help="the ONNX model file")
    args = parser.parse_args(args)

    convert_to_onnx(args.keras_model, args.onnx_model)


if __name__ == "__main__":
    main()

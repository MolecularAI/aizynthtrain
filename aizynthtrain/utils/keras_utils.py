"""Module containing utility classes and routines used in training of policies"""
import functools
import logging
import os
from typing import List, Any, Tuple

import numpy as np
import tensorflow
from tensorflow.config import list_physical_devices
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import (
    EarlyStopping,
    CSVLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import top_k_categorical_accuracy
from scipy import sparse

# Suppress tensforflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf_logger = tensorflow.get_logger()
tf_logger.setLevel(logging.WARNING)

top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
top10_acc.__name__ = "top10_acc"  # type: ignore

top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
top50_acc.__name__ = "top50_acc"  # type: ignore


class InMemorySequence(Sequence):  # pylint: disable=W0223
    """
    Class for in-memory data management

    :param input_filname: the path to the model input data
    :param output_filename: the path to the model output data
    :param batch_size: the size of the batches
    """

    def __init__(
        self, input_filename: str, output_filename: str, batch_size: int
    ) -> None:
        self.batch_size = batch_size
        self.input_matrix = self._load_data(input_filename)
        self.label_matrix = self._load_data(output_filename)
        self.input_dim = self.input_matrix.shape[1]

    def __len__(self) -> int:
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def _make_slice(self, idx: int) -> slice:
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return slice(start, end)

    @staticmethod
    def _load_data(filename: str) -> np.ndarray:
        try:
            return sparse.load_npz(filename)
        except ValueError:
            return np.load(filename)["arr_0"]


def setup_callbacks(
    log_filename: str, checkpoint_filename: str
) -> Tuple[EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau]:
    """
    Setup Keras callback functions: early stopping, CSV logger, model checkpointing,
    and reduce LR on plateau

    :param log_filename: the filename of the CSV log
    :param checkpoint_filename: the filename of the checkpoint
    :return: all the callbacks
    """
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    csv_logger = CSVLogger(log_filename)
    checkpoint = ModelCheckpoint(
        checkpoint_filename,
        monitor="loss",
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=0,
        mode="auto",
        min_delta=0.000001,
        cooldown=0,
        min_lr=0,
    )
    return [early_stopping, csv_logger, checkpoint, reduce_lr]


def train_keras_model(
    model: Model,
    train_seq: InMemorySequence,
    valid_seq: InMemorySequence,
    loss: str,
    metrics: List[Any],
    callbacks: List[Any],
    epochs: int,
) -> None:
    """
    Train a Keras model, but first compiling it and then fitting
    it to the given data

    :param model: the initialized model
    :param train_seq: the training data
    :param valid_seq: the validation data
    :param loss: the loss function
    :param metrics: the metric functions
    :param callbacks: the callback functions
    :param epochs: the number of epochs to use
    """
    print(f"Available GPUs: {list_physical_devices('GPU')}")
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    model.compile(
        optimizer=adam,
        loss=loss,
        metrics=metrics,
    )

    model.fit(
        train_seq,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=valid_seq,
        max_queue_size=20,
        workers=1,
        use_multiprocessing=False,
    )

"""Module routines for training an expansion model"""
import argparse
from typing import Sequence, Optional, Tuple

import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.utils import shuffle


from aizynthtrain.utils.configs import (
    ExpansionModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.keras_utils import (
    InMemorySequence,
    setup_callbacks,
    train_keras_model,
    top10_acc,
    top50_acc,
)


class ExpansionModelSequence(InMemorySequence):
    """
    Custom sequence class to keep sparse, pre-computed matrices in memory.
    Batches are created dynamically by slicing the in-memory arrays
    The data will be shuffled on each epoch end

    :ivar output_dim: the output size (number of templates)
    """

    def __init__(
        self, input_filename: str, output_filename: str, batch_size: int
    ) -> None:
        super().__init__(input_filename, output_filename, batch_size)
        self.output_dim = self.label_matrix.shape[1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        idx_ = self._make_slice(idx)
        return self.input_matrix[idx_].toarray(), self.label_matrix[idx_].toarray()

    def on_epoch_end(self) -> None:
        self.input_matrix, self.label_matrix = shuffle(
            self.input_matrix, self.label_matrix, random_state=0
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line tool for training the model"""
    parser = argparse.ArgumentParser("Tool to training an expansion network policy")
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: ExpansionModelPipelineConfig = load_config(
        args.config, "expansion_model_pipeline"
    )

    train_seq = ExpansionModelSequence(
        config.filename("model_inputs", "training"),
        config.filename("model_labels", "training"),
        config.model_hyperparams.batch_size,
    )
    valid_seq = ExpansionModelSequence(
        config.filename("model_inputs", "validation"),
        config.filename("model_labels", "validation"),
        config.model_hyperparams.batch_size,
    )

    model = Sequential()
    model.add(
        Dense(
            config.model_hyperparams.hidden_nodes,
            input_shape=(train_seq.input_dim,),
            activation="elu",
            kernel_regularizer=regularizers.l2(0.001),
        )
    )
    model.add(Dropout(config.model_hyperparams.dropout))
    model.add(Dense(train_seq.output_dim, activation="softmax"))

    callbacks = setup_callbacks(
        config.filename("training_log"), config.filename("training_checkpoint")
    )

    train_keras_model(
        model,
        train_seq,
        valid_seq,
        "categorical_crossentropy",
        ["accuracy", "top_k_categorical_accuracy", top10_acc, top50_acc],
        callbacks,
        config.model_hyperparams.epochs,
    )


if __name__ == "__main__":
    main()

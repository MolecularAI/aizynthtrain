"""Module routines for training an filter model"""
import argparse
from typing import Sequence, Optional, Tuple, List

import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Dot
from tensorflow.keras.models import Model
from sklearn.utils import shuffle


from aizynthtrain.utils.configs import (
    FilterModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.keras_utils import (
    InMemorySequence,
    setup_callbacks,
    train_keras_model,
)


class FilterModelSequence(InMemorySequence):
    """
    Custom sequence class to keep sparse, pre-computed matrices in memory.
    Batches are created dynamically by slicing the in-memory arrays
    The data will be shuffled on each epoch end
    """

    def __init__(
        self,
        input_filename_rxn: str,
        input_filename_prod: str,
        output_filename: str,
        batch_size: int,
    ) -> None:
        super().__init__(input_filename_prod, output_filename, batch_size)
        self.input_matrix2 = self._load_data(input_filename_rxn)

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], np.ndarray]:
        idx_ = self._make_slice(idx)
        return (
            [self.input_matrix[idx_].toarray(), self.input_matrix2[idx_].toarray()],
            self.label_matrix[idx_],
        )

    def on_epoch_end(self) -> None:
        self.input_matrix, self.input_matrix2, self.label_matrix = shuffle(
            self.input_matrix, self.input_matrix2, self.label_matrix, random_state=0
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    """Command-line tool for training the model"""
    parser = argparse.ArgumentParser("Tool to training an filter network policy")
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: FilterModelPipelineConfig = load_config(
        args.config, "filter_model_pipeline"
    )

    train_seq = FilterModelSequence(
        config.filename("model_inputs_rxn", "training"),
        config.filename("model_inputs_prod", "training"),
        config.filename("model_labels", "training"),
        config.model_hyperparams.batch_size,
    )
    valid_seq = FilterModelSequence(
        config.filename("model_inputs_rxn", "validation"),
        config.filename("model_inputs_prod", "validation"),
        config.filename("model_labels", "validation"),
        config.model_hyperparams.batch_size,
    )

    product_input_layer = Input(shape=(config.model_hyperparams.fingerprint_length,))
    product_dense_layer = Dense(
        config.model_hyperparams.hidden_nodes, activation="elu"
    )(product_input_layer)
    product_droput_layer = Dropout(config.model_hyperparams.dropout)(
        product_dense_layer
    )
    reaction_input_layer = Input(shape=(config.model_hyperparams.fingerprint_length,))
    reaction_dense_layer = Dense(
        config.model_hyperparams.hidden_nodes, activation="elu"
    )(reaction_input_layer)
    cosine_layer = Dot(-1, normalize=True)([product_droput_layer, reaction_dense_layer])
    output_layer = Dense(1, activation="sigmoid")(cosine_layer)
    model = Model(
        inputs=[product_input_layer, reaction_input_layer], outputs=output_layer
    )

    callbacks = setup_callbacks(
        config.filename("training_log"), config.filename("training_checkpoint")
    )

    train_keras_model(
        model,
        train_seq,
        valid_seq,
        "binary_crossentropy",
        ["accuracy"],
        callbacks,
        config.model_hyperparams.epochs,
    )


if __name__ == "__main__":
    main()

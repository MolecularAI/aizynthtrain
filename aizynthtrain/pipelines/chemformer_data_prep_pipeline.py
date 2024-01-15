"""Module containing preparation of a dataset for Chemformer training / evaluation"""
from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from rxnutils.pipeline.runner import main as pipeline_runner
from rxnutils.data.batch_utils import create_csv_batches, combine_csv_batches

from aizynthtrain.modelling.chemformer.create_dataset_split import (
    main as create_dataset_split,
)
from aizynthtrain.utils.configs import ChemformerDataPrepConfig, load_config


class ChemformerDataPrepFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config: ChemformerDataPrepConfig = load_config(
            self.config_path, "chemformer_data_prep"
        )
        self.next(self.reaction_components_setup)

    @step
    def reaction_components_setup(self):
        """Preparing splits of the reaction data for extracting reaction components"""
        self.reaction_partitions = self._create_batches(
            self.config.selected_reactions_path, self.config.reaction_components_path
        )
        self.next(self.reaction_components, foreach="reaction_partitions")

    @step
    def reaction_components(self):
        """
        Running pipeline for specific steps:
            1. Extracting reaction components (reactants / reagents / product)
            2. Removing atom-mapping from components.
        """
        pipeline_path = str(
            Path(__file__).parent / "data" / "chemformer_data_prep_pipeline.yaml"
        )
        idx, start, end = self.input
        if idx > -1:
            pipeline_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    self.config.selected_reactions_path,
                    "--output",
                    f"{self.config.reaction_components_path}.{idx}",
                    "--max-workers",
                    "1",
                    "--batch",
                    str(start),
                    str(end),
                    "--no-intermediates",
                ]
            )
        self.next(self.reaction_components_join)

    @step
    def reaction_components_join(self, inputs):
        """Joining split reactions"""
        self.config = inputs[0].config
        self._combine_batches(self.config.reaction_components_path)
        self.next(self.dataset_split)

    @step
    def dataset_split(self):
        """Splitting data into train / validation / test sets"""
        create_dataset_split(
            [
                "--config_path",
                self.config_path,
            ]
        )
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Processed Chemformer dataset is located here: {self.config.chemformer_data_path}"
        )

    def _combine_batches(self, filename):
        combine_csv_batches(filename, self.config.nbatches)

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.nbatches, output_filename)


if __name__ == "__main__":
    ChemformerDataPrepFlow()

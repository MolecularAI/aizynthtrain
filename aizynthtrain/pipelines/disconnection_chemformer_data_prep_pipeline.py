"""Module containing preparation of a dataset for Chemformer training / evaluation"""
from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from rxnutils.pipeline.runner import main as pipeline_runner

from aizynthtrain.utils.configs import ChemformerDataPrepConfig, load_config
from rxnutils.data.batch_utils import combine_csv_batches, create_csv_batches


class DisconnectionChemformerDataPrepFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config: ChemformerDataPrepConfig = load_config(
            self.config_path, "chemformer_data_prep"
        )
        self.next(self.tag_products_setup)

    @step
    def tag_products_setup(self):
        """Preparing splits of the preprocessed reaction data for tagging disconnection sites."""
        self.reaction_partitions = self._create_batches(
            self.config.chemformer_data_path, self.config.disconnection_aware_data_path
        )
        self.next(self.tag_products, foreach="reaction_partitions")

    @step
    def tag_products(self):
        """
        Running pipeline for tagging disconnection sites in products.
        """
        pipeline_path = str(
            Path(__file__).parent / "data" / "chemformer_product_tagging_pipeline.yaml"
        )
        idx, start, end = self.input
        if idx > -1:
            pipeline_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    self.config.chemformer_data_path,
                    "--output",
                    f"{self.config.disconnection_aware_data_path}.{idx}",
                    "--max-workers",
                    "1",
                    "--batch",
                    str(start),
                    str(end),
                    "--no-intermediates",
                ]
            )
        self.next(self.tag_products_join)

    @step
    def tag_products_join(self, inputs):
        """Joining split reactions"""
        self.config = inputs[0].config
        self._combine_batches(self.config.disconnection_aware_data_path)
        self.next(self.create_autotag_dataset_setup)

    @step
    def create_autotag_dataset_setup(self):
        """Preparing splits of the preprocessed reaction data for tagging disconnection sites."""
        self.reaction_partitions = self._create_batches(
            self.config.disconnection_aware_data_path, self.config.autotag_data_path
        )
        self.next(self.create_autotag_dataset, foreach="reaction_partitions")

    @step
    def create_autotag_dataset(self):

        pipeline_path = str(
            Path(__file__).parent / "data" / "autotag_data_extraction.yaml"
        )
        idx, start, end = self.input
        if idx > -1:
            pipeline_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    self.config.disconnection_aware_data_path,
                    "--output",
                    f"{self.config.autotag_data_path}.{idx}",
                    "--max-workers",
                    "1",
                    "--no-intermediates",
                    "--batch",
                    str(start),
                    str(end),
                ]
            )
        self.next(self.create_autotag_dataset_join)

    @step
    def create_autotag_dataset_join(self, inputs):
        """Joining split reactions"""
        self.config = inputs[0].config
        self._combine_batches(self.config.autotag_data_path)
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Disconnection-aware dataset is located here: {self.config.disconnection_aware_data_path}"
        )
        print(f"Autotag dataset is located here: {self.config.autotag_data_path}")

    def _combine_batches(self, filename):
        combine_csv_batches(filename, self.config.nbatches)

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.nbatches, output_filename)


if __name__ == "__main__":
    DisconnectionChemformerDataPrepFlow()

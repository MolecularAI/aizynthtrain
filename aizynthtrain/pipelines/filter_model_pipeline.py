"""Module containing a pipeline for training a filter model"""
from pathlib import Path
from re import I

from metaflow import FlowSpec, Parameter, step

from aizynthtrain.modelling.filter_policy.create_metadata import (
    main as create_metadata,
)
from aizynthtrain.modelling.filter_policy.featurize import (
    main as featurize,
)
from aizynthtrain.modelling.filter_policy.split_data import (
    main as split_data,
)
from aizynthtrain.modelling.filter_policy.training import (
    main as training_runner,
)
from aizynthtrain.utils.reporting import main as report_runner
from aizynthtrain.utils.files import (
    create_csv_batches,
    combine_sparse_matrix_batches,
    combine_numpy_array_batches,
)
from aizynthtrain.utils.configs import (
    FilterModelPipelineConfig,
    load_config,
)


class FilterModelFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config: FilterModelPipelineConfig = load_config(
            self.config_path, "filter_model_pipeline"
        )
        self.next(self.create_metadata)

    @step
    def create_metadata(self):
        """Preprocess the library for model training"""
        if not Path(self.config.filename("split_indices")).exists():
            create_metadata([self.config_path])
        self.next(self.featurization_setup)

    @step
    def featurization_setup(self):
        """Preparing splits of the reaction data for feauturization"""
        self.reaction_partitions = self._create_batches(
            self.config.filename("library"), self.config.filename("model_labels")
        )
        self.next(self.featurization, foreach="reaction_partitions")

    @step
    def featurization(self):
        """Featurization of reaction data"""
        idx, start, end = self.input
        if idx > -1:
            featurize(
                [
                    self.config_path,
                    "--batch",
                    str(idx),
                    str(start),
                    str(end),
                ]
            )
        self.next(self.featurization_join)

    @step
    def featurization_join(self, inputs):
        """Joining featurized data"""
        self.config = inputs[0].config
        self._combine_batches(self.config.filename("model_labels"), is_sparse=False)
        self._combine_batches(self.config.filename("model_inputs_rxn"), is_sparse=True)
        self._combine_batches(self.config.filename("model_inputs_prod"), is_sparse=True)
        self.next(self.split_data)

    @step
    def split_data(self):
        """Split featurized data"""
        if not Path(self.config.filename("model_labels", "training")).exists():
            split_data([self.config_path])
        self.next(self.model_training)

    @step
    def model_training(self):
        """Train the filter model"""
        if not Path(self.config.filename("training_checkpoint")).exists():
            training_runner([self.config_path])
        self.next(self.report_creation)

    @step
    def report_creation(self):
        """Create report of trained model"""
        notebook_path = str(Path(__file__).parent / "notebooks" / "filter_model_val.py")
        report_runner(
            [
                "--notebook",
                notebook_path,
                "--report_path",
                self.config.filename("report"),
                "--python_kernel",
                self.config.python_kernel,
                "--validation_metrics_filename",
                self.config.filename("training_log"),
            ]
        )
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Report on trained model is located here: {self.config.filename('report')}"
        )

    def _combine_batches(self, filename, is_sparse):
        if Path(filename).exists():
            return
        if is_sparse:
            combine_sparse_matrix_batches(filename, self.config.nbatches)
        else:
            combine_numpy_array_batches(filename, self.config.nbatches)

    def _create_batches(self, input_filename, output_filename):
        if Path(output_filename).exists():
            return [(-1, None, None)]
        return create_csv_batches(input_filename, self.config.nbatches)


if __name__ == "__main__":
    FilterModelFlow()

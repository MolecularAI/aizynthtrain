"""Module containing a pipeline for training an expansion model"""
from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from rxnutils.data.batch_utils import combine_sparse_matrix_batches, create_csv_batches

from aizynthtrain.utils.onnx_converter import main as convert_to_onnx
from aizynthtrain.modelling.expansion_policy.create_template_lib import (
    main as create_template_lib,
)
from aizynthtrain.modelling.expansion_policy.eval_multi_step import (
    main as eval_multi_step,
)
from aizynthtrain.modelling.expansion_policy.eval_one_step import main as eval_one_step
from aizynthtrain.modelling.expansion_policy.featurize import main as featurize
from aizynthtrain.modelling.expansion_policy.split_data import main as split_data
from aizynthtrain.modelling.expansion_policy.training import main as training_runner
from aizynthtrain.utils.configs import ExpansionModelPipelineConfig, load_config
from aizynthtrain.utils.reporting import main as report_runner


class ExpansionModelFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config: ExpansionModelPipelineConfig = load_config(
            self.config_path, "expansion_model_pipeline"
        )
        self.next(self.create_template_metadata)

    @step
    def create_template_metadata(self):
        """Preprocess the template library for model training"""
        if not Path(self.config.filename("unique_templates")).exists():
            create_template_lib([self.config_path])
        self.next(self.featurization_setup)

    @step
    def featurization_setup(self):
        """Preparing splits of the reaction data for feauturization"""
        self.reaction_partitions = self._create_batches(
            self.config.filename("template_code"), self.config.filename("model_labels")
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
        self._combine_batches(self.config.filename("model_labels"))
        self._combine_batches(self.config.filename("model_inputs"))
        self.next(self.split_data)

    @step
    def split_data(self):
        """Split featurized data into training, validation and testing"""
        if not Path(self.config.filename("model_labels", "training")).exists():
            split_data([self.config_path])
        self.next(self.model_training)

    @step
    def model_training(self):
        """Train the expansion model"""
        if not Path(self.config.filename("training_checkpoint")).exists():
            training_runner([self.config_path])
        self.next(self.onnx_converter)

    @step
    def onnx_converter(self):
        """Convert the trained Keras model to ONNX"""
        convert_to_onnx(
            [
                self.config.filename("training_checkpoint"),
                self.config.filename("onnx_model"),
            ]
        )
        self.next(self.model_validation)

    @step
    def model_validation(self):
        """Validate the trained model"""
        eval_one_step([self.config_path])
        eval_multi_step([self.config_path])
        notebook_path = str(
            Path(__file__).parent / "notebooks" / "expansion_model_val.py"
        )
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
                "--onestep_report",
                self.config.filename("onestep_report"),
                "--multistep_report",
                self.config.filename("multistep_report"),
            ]
        )
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Report on trained model is located here: {self.config.filename('report')}"
        )

    def _combine_batches(self, filename):
        combine_sparse_matrix_batches(filename, self.config.nbatches)

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.nbatches, output_filename)


if __name__ == "__main__":
    ExpansionModelFlow()

"""Module containing preparing a template library"""
from pathlib import Path

from metaflow import FlowSpec, Parameter, step
from rxnutils.pipeline.runner import main as validation_runner
from rxnutils.data.batch_utils import create_csv_batches, combine_csv_batches

from aizynthtrain.utils.configs import (
    load_config,
    TemplatePipelineConfig,
)
from aizynthtrain.utils.reporting import main as selection_runner
from aizynthtrain.utils.template_runner import main as template_runner
from aizynthtrain.utils.stereo_flipper import main as stereo_flipper
from aizynthtrain.utils.files import prefix_filename


class StereoTemplatesExtractionFlow(FlowSpec):
    config_path = Parameter("config", required=True)

    @step
    def start(self):
        """Loading configuration"""
        self.config: TemplatePipelineConfig = load_config(
            self.config_path, "template_pipeline"
        )
        self.next(self.stereo_checks_setup)

    @step
    def stereo_checks_setup(self):
        """Preparing splits of the reaction data for stereo checks"""
        self.reaction_partitions = self._create_batches(
            self.config.selected_reactions_path.replace(".csv", "_all.csv"),
            self.config.stereo_reactions_path,
        )
        self.next(self.stereo_checks, foreach="reaction_partitions")

    @step
    def stereo_checks(self):
        """Validating reaction data"""
        pipeline_path = str(
            Path(__file__).parent / "data" / "stereo_checks_pipeline.yaml"
        )
        idx, start, end = self.input
        batch_output = f"{self.config.stereo_reactions_path}.{idx}"
        if idx > -1 and not Path(batch_output).exists():
            validation_runner(
                [
                    "--pipeline",
                    pipeline_path,
                    "--data",
                    self.config.selected_reactions_path.replace(".csv", "_all.csv"),
                    "--output",
                    batch_output,
                    "--max-workers",
                    "1",
                    "--batch",
                    str(start),
                    str(end),
                    "--no-intermediates",
                ]
            )
        else:
            print(
                f"Skipping template extraction for idx {idx}. File exists = {Path(batch_output).exists()}"
            )
        self.next(self.stereo_checks_join)

    @step
    def stereo_checks_join(self, inputs):
        """Joining stereo reactions"""
        self.config = inputs[0].config
        self._combine_batches(self.config.stereo_reactions_path)
        self.next(self.stereo_selection)

    @step
    def stereo_selection(self):
        """Selecting stereo reactions and producing report"""
        notebook_path = str(Path(__file__).parent / "notebooks" / "stereo_selection.py")
        if not Path(self.config.selected_stereo_reactions_path).exists():
            selection_runner(
                [
                    "--notebook",
                    notebook_path,
                    "--report_path",
                    self.config.stereo_report_path,
                    "--python_kernel",
                    self.config.python_kernel,
                    "--input_filename",
                    self.config.stereo_reactions_path,
                    "--output_filename",
                    self.config.selected_stereo_reactions_path,
                ]
            )
        self.next(self.template_extraction_setup)

    @step
    def template_extraction_setup(self):
        """Preparing splits of the reaction data for template extraction"""
        self.reaction_partitions = self._create_batches(
            self.config.selected_stereo_reactions_path,
            self.config.unvalidated_templates_path,
        )
        self.next(self.template_extraction, foreach="reaction_partitions")

    @step
    def template_extraction(self):
        """Extracting RDChiral reaction templates"""
        idx, start, end = self.input
        batch_output = f"{self.config.unvalidated_templates_path}.{idx}"
        if idx > -1 and not Path(batch_output).exists():
            template_runner(
                [
                    "--input_path",
                    self.config.selected_stereo_reactions_path,
                    "--output_path",
                    batch_output,
                    "--radius",
                    "1",
                    "--smiles_column",
                    "RxnSmilesClean",
                    "--ringbreaker_column",
                    "RingBreaker",
                    "--batch",
                    str(start),
                    str(end),
                ]
            )
        else:
            print(
                f"Skipping template extraction for idx {idx}. File exists = {Path(batch_output).exists()}"
            )
        self.next(self.template_extraction_join)

    @step
    def template_extraction_join(self, inputs):
        """Joining extracted templates"""
        self.config = inputs[0].config
        self._combine_batches(self.config.unvalidated_templates_path)
        self.next(self.template_validation_setup)

    @step
    def template_validation_setup(self):
        """Preparing splits of the reaction data for template validation"""
        self.reaction_partitions = self._create_batches(
            self.config.unvalidated_templates_path, self.config.validated_templates_path
        )
        self.next(self.template_validation, foreach="reaction_partitions")

    @step
    def template_validation(self):
        """Validating extracted templates"""
        pipline_path = str(
            Path(__file__).parent / "data" / "template_validation_pipeline.yaml"
        )
        idx, start, end = self.input
        if idx > -1:
            validation_runner(
                [
                    "--pipeline",
                    pipline_path,
                    "--data",
                    self.config.unvalidated_templates_path,
                    "--output",
                    f"{self.config.validated_templates_path}.{idx}",
                    "--max-workers",
                    "1",
                    "--batch",
                    str(start),
                    str(end),
                    "--no-intermediates",
                ]
            )
            stereo_flipper(
                [
                    "--input_path",
                    f"{self.config.validated_templates_path}.{idx}",
                    "--query",
                    "StereoBucket=='reagent controlled'",
                ]
            )
        self.next(self.template_validation_join)

    @step
    def template_validation_join(self, inputs):
        """Joining validated templates"""
        self.config = inputs[0].config
        self._combine_batches(self.config.validated_templates_path)
        self.next(self.template_selection)

    @step
    def template_selection(self):
        """Selection templates and produce report"""
        notebook_path = str(
            Path(__file__).parent / "notebooks" / "stereo_template_selection.py"
        )
        output_path = prefix_filename(
            self.config.selected_templates_prefix,
            self.config.selected_templates_postfix,
        )
        if not Path(output_path).exists():
            selection_runner(
                [
                    "--notebook",
                    notebook_path,
                    "--report_path",
                    self.config.templates_report_path,
                    "--python_kernel",
                    self.config.python_kernel,
                    "--input_filename",
                    self.config.validated_templates_path,
                    "--output_prefix",
                    self.config.selected_templates_prefix,
                    "--output_postfix",
                    self.config.selected_templates_postfix,
                    "--min_occurrence",
                    str(self.config.min_template_occurrence),
                    "--config_filename",
                    self.config_path,
                ]
            )
        self.next(self.end)

    @step
    def end(self):
        print(
            f"Report on extracted reaction is located here: {self.config.stereo_report_path}"
        )
        print(
            f"Report on extracted templates is located here: {self.config.templates_report_path}"
        )

    def _combine_batches(self, filename):
        combine_csv_batches(filename, self.config.nbatches)

    def _create_batches(self, input_filename, output_filename):
        return create_csv_batches(input_filename, self.config.nbatches, output_filename)


if __name__ == "__main__":
    StereoTemplatesExtractionFlow()

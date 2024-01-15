"""Module containing configuration classes"""
import sys
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


def load_config(filename: str, class_key: str) -> Any:
    """
    Loading any configuration class from a YAML file

    The `class_key` should be a snake-case version of class name
    in this module, excluding the "Config" string.
    For example: template_pipeline -> TemplatePipelineConfig

    :param filename: the path to the YAML file
    :param class_key: the name of the configuration at the top-level
    """
    with open(filename, "r") as fileobj:
        dict_ = yaml.load(fileobj.read(), Loader=yaml.SafeLoader)

    module = sys.modules[__name__]
    class_name = "".join(part.title() for part in class_key.split("_")) + "Config"
    if not hasattr(module, class_name):
        raise KeyError(f"No config class matches {class_key}")
    class_ = getattr(module, class_name)
    try:
        config = class_(**dict_.get(class_key, {}))
    except TypeError as err:
        raise ValueError(f"Unable to setup config class with message: {err}")

    return config


class _FilenameMixin:
    """Mixin-class for providing easy access to filenames"""

    def filename(self, name: str, subset: str = "") -> str:
        name_value = getattr(self.postfixes, name)
        if subset:
            name_value = subset + "_" + name_value
        if self.file_prefix:
            return self.file_prefix + "_" + name_value
        return name_value


class TemplatePipelineConfig(BaseModel):
    """Configuration class for template pipeline"""

    python_kernel: str
    data_import_class: str
    data_import_config: Dict[str, str] = Field(default_factory=dict)
    selected_templates_prefix: str = ""
    selected_templates_postfix: str = "template_library.csv"
    import_data_path: str = "imported_reactions.csv"
    validated_reactions_path: str = "validated_reactions.csv"
    selected_reactions_path: str = "selected_reactions.csv"
    reaction_report_path: str = "reaction_selection_report.html"
    stereo_reactions_path: str = "stereo_reactions.csv"
    selected_stereo_reactions_path: str = "selected_stereo_reactions.csv"
    stereo_report_path: str = "stereo_selection_report.html"
    unvalidated_templates_path: str = "reaction_templates_unvalidated.csv"
    validated_templates_path: str = "reaction_templates_validated.csv"
    templates_report_path: str = "template_selection_report.html"
    min_template_occurrence: int = 10
    nbatches: int = 200


# Config classes for the chemformer model pipelines


class ChemformerTrainConfig(BaseModel):
    """Configuration class for chemformer training"""

    data_path: str = "proc_selected_reactions.csv"
    vocabulary_path: str = "bart_vocab_downstream.json"
    task: str = "backward_prediction"
    model_path: Optional[str] = None
    deepspeed_config_path: str = "ds_config.json"
    output_directory: str = ""

    n_epochs: int = 70
    n_gpus: int = 1
    batch_size: int = 64
    augmentation_probability: float = 0.5
    learning_rate: float = 0.001

    check_val_every_n_epoch: int = 1
    checkpoint_every_n_step: int = 25000
    limit_val_batches: float = 1.0
    augmentation_strategy: str = "all"
    n_nodes: int = 1
    acc_batches: int = 4
    schedule: str = "cycle"
    dataset_type: str = "synthesis"

    # Arguments for model from random initialization
    d_model: int = 512
    n_layer: int = 6
    n_heads: int = 8
    d_feedforward: int = 2048


class ChemformerDataPrepConfig(BaseModel):
    """Configuration class for chemformer data-prep pipeline"""

    selected_reactions_path: str = "selected_reactions.csv"
    reaction_components_path: str = "reaction_components.csv"
    routes_to_exclude: List[str] = Field(default_factory=list)
    training_fraction: float = 0.9
    random_seed: int = 11
    nbatches: int = 200
    chemformer_data_path: str = "proc_selected_reactions.csv"

    reaction_hash_col: str = "reaction_hash"
    set_col: str = "set"
    is_external_col: str = "is_external"


# Config classes for the expansion model pipeline


class TemplateLibraryColumnsConfig(BaseModel):
    """Configuration class for columns in a template library file"""

    reaction_smiles: str = "reaction_smiles"
    reaction_hash: str = "reaction_hash"
    retro_template: str = "retro_template"
    template_hash: str = "template_hash"
    template_code: str = "template_code"
    library_occurrence: str = "library_occurence"
    classification: Optional[str] = "classification"
    ring_breaker: Optional[str] = "ring_breaker"
    stereo_bucket: Optional[str] = "stereo_bucket"
    flipped_stereo: Optional[str] = "flipped_stereo"


class TemplateLibraryConfig(BaseModel):
    """Configuration class for template library generation"""

    columns: TemplateLibraryColumnsConfig = Field(
        default_factory=TemplateLibraryColumnsConfig
    )
    metadata_columns: List[str] = Field(
        default_factory=lambda: ["template_hash", "classification"]
    )
    template_set: str = "templates"


class ExpansionModelPipelinePostfixes(BaseModel):
    """Configuration class for postfixes of files generated by expansion model pipeline"""

    library: str = "template_library.csv"
    template_code: str = "template_code.csv"
    unique_templates: str = "unique_templates.csv.gz"
    template_lookup: str = "lookup.json"
    split_indices: str = "split_indices.npz"
    model_labels: str = "labels.npz"
    model_inputs: str = "inputs.npz"
    training_log: str = "keras_training.csv"
    training_checkpoint: str = "keras_model.hdf5"
    onnx_model: str = "expansion.onnx"

    report: str = "expansion_model_report.html"
    finder_output: str = "model_validation_finder_output.hdf5"
    multistep_report: str = "model_validation_multistep_report.json"
    expander_output: str = "model_validation_expander_output.json"
    onestep_report: str = "model_validation_onestep_report.json"


class ExpansionModelHyperparamsConfig(BaseModel):
    """Configuration class for hyper parameters for expansion model training"""

    fingerprint_radius: int = 2
    fingerprint_length: int = 2048
    batch_size: int = 256
    hidden_nodes: int = 512
    dropout: int = 0.4
    epochs: int = 100
    chirality: bool = False


class ExpansionModelPipelineConfig(BaseModel, _FilenameMixin):
    """Configuration class for expansion model pipeline"""

    python_kernel: str
    file_prefix: str = ""
    nbatches: int = 200
    training_fraction: float = 0.9
    random_seed: int = 1689
    selected_ids_path: str = "selected_reactions_ids.json"
    routes_to_exclude: List[str] = Field(default_factory=list)
    library_config: TemplateLibraryConfig = Field(default_factory=TemplateLibraryConfig)
    postfixes: ExpansionModelPipelinePostfixes = Field(
        default_factory=ExpansionModelPipelinePostfixes
    )
    model_hyperparams: ExpansionModelHyperparamsConfig = Field(
        default_factory=ExpansionModelHyperparamsConfig
    )


class ExpansionModelEvaluationConfig(BaseModel, _FilenameMixin):
    """Configuration class for evaluation of expansion model"""

    stock_for_finding: str = ""
    stock_for_recovery: str = ""
    search_properties_for_finding: Dict[str, Any] = Field(
        default_factory=lambda: {"return_first": True}
    )
    search_properties_for_recovery: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_transforms": 10,
            "iteration_limit": 500,
            "time_limit": 3600,
        }
    )
    reference_routes: str = ""
    target_smiles: str = ""
    file_prefix: str = ""
    top_n: int = 50
    n_test_reactions: int = 1000
    columns: TemplateLibraryColumnsConfig = Field(
        default_factory=TemplateLibraryColumnsConfig
    )
    postfixes: ExpansionModelPipelinePostfixes = Field(
        default_factory=ExpansionModelPipelinePostfixes
    )

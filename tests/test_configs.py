import yaml
import pytest

from aizynthtrain.utils.configs import (
    load_config,
    TemplatePipelineConfig,
    ExpansionModelPipelineConfig,
    TemplateLibraryConfig,
    ExpansionModelHyperparamsConfig,
    ExpansionModelEvaluationConfig,
)


@pytest.fixture
def make_config_file(tmpdir):
    filename = str(tmpdir / "config.yml")

    def wrapper(config_dict):
        with open(filename, "w") as fileobj:
            yaml.dump(config_dict, fileobj)
        return filename

    return wrapper


def test_template_pipeline_config(make_config_file):
    dict_ = {
        "python_kernel": "aizynth",
        "data_import_class": "aizynthtrain.data.uspto.importer.UsptoImporter",
    }
    filename = make_config_file({"template_pipeline": dict_})

    config = load_config(filename, "template_pipeline")

    assert type(config) == TemplatePipelineConfig
    assert config.python_kernel == dict_["python_kernel"]
    assert config.data_import_class == dict_["data_import_class"]


def test_expansion_model_pipeline_config(make_config_file):
    dict_ = {
        "python_kernel": "aizynth",
        "file_prefix": "uspto",
        "routes_to_exclude": ["ref_routes_n1.json", "ref_routes_n5.json"],
        "library_config": {"template_set": "uspto_templates"},
        "postfixes": {"report": "model_report.html"},
        "model_hyperparams": {"epochs": 50},
    }
    filename = make_config_file({"expansion_model_pipeline": dict_})

    config = load_config(filename, "expansion_model_pipeline")

    assert type(config) == ExpansionModelPipelineConfig
    assert config.python_kernel == dict_["python_kernel"]
    assert config.file_prefix == dict_["file_prefix"]
    assert len(config.routes_to_exclude) == 2

    assert type(config.library_config) == TemplateLibraryConfig
    assert config.library_config.template_set == dict_["library_config"]["template_set"]

    assert type(config.model_hyperparams) == ExpansionModelHyperparamsConfig
    assert config.model_hyperparams.epochs == 50

    assert config.filename("report") == "uspto_model_report.html"
    assert config.filename("model_labels", "training") == "uspto_training_labels.npz"


def test_expansion_model_evaluation_config(make_config_file):
    dict_ = {
        "file_prefix": "uspto",
        "stock_for_finding": "stock_for_eval_find.hdf5",
        "target_smiles": "target_smiles",
        "search_properties_for_finding": {},
    }
    filename = make_config_file({"expansion_model_evaluation": dict_})

    config = load_config(filename, "expansion_model_evaluation")

    assert type(config) == ExpansionModelEvaluationConfig
    assert config.file_prefix == dict_["file_prefix"]
    assert config.stock_for_finding == dict_["stock_for_finding"]
    assert config.target_smiles == dict_["target_smiles"]
    assert config.search_properties_for_finding == {}

    assert (
        config.filename("multistep_report")
        == "uspto_model_validation_multistep_report.json"
    )

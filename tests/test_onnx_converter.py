"""Tests the utils module to convert Keras models to ONNX."""

import pytest
import pytest_mock

from aizynthtrain.utils import onnx_converter


@pytest.fixture
def mock_tf2onnx_convert_from_keras(mocker: pytest_mock.MockerFixture):
    return mocker.patch(
        "aizynthtrain.utils.onnx_converter.tf2onnx.convert.from_keras",
        return_value=None,
    )
    

@pytest.fixture
def mock_load_keras_model(mocker: pytest_mock.MockerFixture):
    return mocker.patch(
        "aizynthtrain.utils.onnx_converter.load_model",
        return_value=None,
    )
    

def test_mock_convert_to_onnx(mock_tf2onnx_convert_from_keras, mock_load_keras_model) -> None:
    test_convert = onnx_converter.convert_to_onnx("test_model_path", "test_onnx_path")
    mock_tf2onnx_convert_from_keras.assert_called_once()
    mock_load_keras_model.assert_called_once()
    
    
def test_convert_to_onnx(tmpdir, shared_datadir) -> None:
    test_convert = onnx_converter.convert_to_onnx(
        shared_datadir / "test_keras.hdf5", 
        str(tmpdir / "test_model.onnx")
    )
    assert (tmpdir / "test_model.onnx").exists()

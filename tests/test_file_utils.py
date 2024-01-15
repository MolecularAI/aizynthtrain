import pytest

from aizynthtrain.utils.files import prefix_filename


@pytest.mark.parametrize(
    ("prefix", "postfix", "expected"),
    [("a", "b", "a_b"), ("", "b", "b"), ("a", "", "a_"), ("", "", "")],
)
def test_prefix_filename(prefix, postfix, expected):
    assert prefix_filename(prefix, postfix) == expected

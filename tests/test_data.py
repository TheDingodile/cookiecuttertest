import os.path

import pytest

from src.data.mnist import load
from tests import _PATH_DATA, _PROJECT_ROOT


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_size():
    input_train, labels_train, test_inputs, test_labels = load(input_path="data/processed")
    assert len(input_train) == 25000
    assert len(labels_train) == 25000
    assert len(test_inputs) == 5000
    assert len(test_labels) == 5000
# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented
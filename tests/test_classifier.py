"""
Testing Classifier module
"""

import pytest
from classification.classifier import Classifier


@pytest.fixture
def setup_pipeline():
    """Setup classifier pipeline - training classifier and saving model"""
    pipeline = Classifier()
    pipeline.train_and_save()
    return pipeline


@pytest.fixture
def requests():
    """Example dataset"""
    return {"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}


@pytest.fixture
def response():
    """Ground truth response from classifier"""
    return ["virginica", "versicolor"]


def test_response(setup_pipeline, requests, response):
    """Tests if classifier returns correct prediction"""
    assert response == setup_pipeline.load_and_test(requests["features"])["predictions"]

import pytest
from classification.classifier import Classifier

@pytest.fixture
def setup_pipeline():
    pipeline = Classifier()
    pipeline.train_and_save()
    return pipeline

@pytest.fixture
def requests():
    return {
        "features": [
            [6.5, 3.0, 5.8, 2.2],
            [6.1, 2.8, 4.7, 1.2]
        ]
    }

@pytest.fixture
def response():
    return ["virginica", "versicolor"]

def test_response(setup_pipeline, requests, response):
    assert response == setup_pipeline.load_and_test(requests)["predictions"]

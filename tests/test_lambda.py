"""
Testing Lambda handler
"""

import os
import sys
import json
import pytest

from lambda_function import lambda_handler


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.dirname(parent_dir))


@pytest.fixture
def event():
    """Example json event"""
    json_event = {"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}
    return json_event


@pytest.fixture
def context():
    """Example context"""
    return None


@pytest.fixture
def response_prediction():
    """Ground truth - API response"""
    return ["virginica", "versicolor"]


def test_lambda_handler(event, context, response_prediction):
    """Tests lambda handler"""
    lambda_response = lambda_handler(event, context)

    assert lambda_response["statusCode"] == 200
    assert json.loads(lambda_response["body"])["predictions"] == response_prediction

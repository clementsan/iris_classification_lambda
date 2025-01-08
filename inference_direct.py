"""
Direct inference with hard-coded data
"""

import json
from classification.classifier import Classifier


if __name__ == "__main__":
    cls = Classifier()

    # Training
    cls.train_and_save()

    # Testing
    data = {"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}
    features = data["features"]
    results = cls.load_and_test(features)
    print("results:", results)

    # Response similar to REST API call
    response = {
        "statusCode": 200,
        "body": json.dumps(
            {"predictions": results["predictions"], "probabilities": results["probabilities"]}
        ),
    }
    print("Example REST API response: ", response)

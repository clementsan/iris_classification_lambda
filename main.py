from classification.classifier import Classifier
import json


if __name__ == "__main__":
    cls = Classifier()

    # Training
    cls.train_and_save()

    # Testing
    data = { "features": [
        [6.5, 3.0, 5.8, 2.2],
        [6.1, 2.8, 4.7, 1.2]
    ]}
    results = cls.load_and_test(data)
    print("results:", results)

    # Response similar to HTTP call
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'predictions': results["predictions"],
            'probabilities': results["probabilities"]
        })
    }
    print("Example Full API response: ", response)
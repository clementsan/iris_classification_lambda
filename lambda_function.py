from classification.classifier import Classifier
import json


cls = Classifier()

# Lambda handler (proxy integration option unchecked on AWS API Gateway)
def lambda_handler(event, context):

    try:
        features = event.get('features', {})
        if not features:
            raise ValueError("'features' key missing")

        response = cls.load_and_test(features)
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'predictions': response["predictions"],
                'probabilities': response["probabilities"]
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

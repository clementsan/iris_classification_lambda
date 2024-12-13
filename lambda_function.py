from classification.classifier import Classifier
import json


cls = Classifier()

def lambda_handler(event, context):
    try:
        # Parse the input data
        data = json.loads(event.get('body', '{}'))

        response = cls.load_and_test(data)
        print("Lambda response: ", response)

        return {
            'statusCode': 200,
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

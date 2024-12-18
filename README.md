---
title: IRIS Classification Lambda
emoji: 🏢
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
short_description: IRIS Classification Lambda
---

# IRIS classification task with AWS Lambda

Workflow: use of AWS lambda function for deployment

## Local development

### Training the model:

bash
> python train.py

### Building the docker image:

bash
> docker build -t iris-classification-lambda .

### Running the docker container locally:

bash

> docker run --name iris-classification-lambda-cont -p 8080:8080 iris-classification-lambda


### Testing locally:

Example of a prediction request

bash
> curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" -H "Content-Type: application/json" -d '{"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}'

python
> python3 inference_api.py --url http://localhost:8080/2015-03-31/functions/function/invocations -d '{"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}'


## Deployment to AWS

### Pushing the docker container to AWS ECR

Steps:
 - Create new ECR Repository via aws console

Example: ```iris-classification-lambda```


 - Optional for aws cli configuration (to run above commands):
> aws configure
 
 - Authenticate Docker client to the Amazon ECR registry
> aws ecr get-login-password --region <aws_region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com

 - Tag local docker image with the Amazon ECR registry and repository
> docker tag iris-classification-lambda:latest <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/iris-classification-lambda:latest

 - Push docker image to ECR
> docker push <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/iris-classification-lambda:latest

[Link to AWS Documention](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)

### Creating and testing a Lambda function

<b>Steps</b>: 
 - Create function from container image

Example name: ```iris-classification```

 - Notes: the API endpoint will use the ```lambda_function.py``` file and ```lambda_hander``` function
 - Test the lambda via the AWS console

Example JSON object:
```
{
    "features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]
}
```

Advanced notes:
 - Steps to update the Lambda function with latest container via aws cli:
> aws lambda update-function-code --function-name iris-classification --image-uri <aws_account_id>.dkr.ecr.<aws_region>.amazonaws.com/iris-classification-lambda:latest


### Creating an API via API Gateway


<b>Steps</b>: 
 - Create a new ```Rest API``` (e.g. ```iris-classification-api```)
 - Add a new resource to the API (e.g. ```/classify```)
 - Add a ```POST``` method to the resource
 - Integrate the Lambda function to the API
   - Notes: using proxy integration option unchecked
 - Deploy API with a specific stage (e.g. ```test``` stage)

Example API Endpoint URL:
https://<api_id>.execute-api.<aws_region>.amazonaws.com/test/classify
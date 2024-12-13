# IRIS classification task with AWS Lambda

Workflow: use of AWS lambda function for deployment

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

Example of a prediction request via curl command

bash
> curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{"body": "{\"features\": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}"}'


### Deployment to AWS

Steps:
 - Pushing the docker container to AWS ECR
 - Creating and testing a Lambda function
 - Creating an API via API Gateway
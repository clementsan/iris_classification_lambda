# IRIS classification task with AWS Lambda

## Workflow: use of AWS lambda function for deployment
Steps to Deploy

### Training the Model:

bash
> python train.py

### Building the docker image:

bash
> docker build -t iris-lambda .

### Running the docker container locally:

bash

> docker run --name iris-lambda-cont -p 8080:8080 iris-lambda


### Testing locally:

Use a tool like curl to send a test request:

bash
> curl -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{"body": "{\"features\": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}"}'

Deploy to AWS Lambda: Package the code and dependencies, then upload to AWS Lambda via the AWS Management Console or AWS CLI.

This setup provides a complete pipeline from training the model to deploying it on AWS Lambda.

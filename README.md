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

<b>Aims:</b> Categorization of different species of iris flowers (Setosa, Versicolor, and Virginica) 
            based on measurements of physical characteristics (sepals and petals).

<b>Architecture:</b>
 - Front-end: user interface via Gradio library
 - Back-end: use of AWS Lambda function to run deployed ML model

You can try out our deployed [Hugging Face Space](https://huggingface.co/spaces/cvachet/iris_classification_lambda
)!

<b>Table of contents: </b>
 - [Local development](#1-local-development)
 - [AWS deployment](#2-deployment-to-aws)
 - [Hugging Face deployment](#3-deployment-to-hugging-face)


## 1. Local development

### 1.1 Training the ML model

bash
> python train.py

### 1.2. Docker container

 - Building the docker image

bash
> docker build -t iris-classification-lambda .

 - Running the docker container

bash

> docker run --name iris-classification-lambda-cont -p 8080:8080 iris-classification-lambda


### 1.3. Execution via command line

Example of a prediction request

bash
> curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" -H "Content-Type: application/json" -d '{"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}'

python
> python3 inference_api.py --url http://localhost:8080/2015-03-31/functions/function/invocations -d '{"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}'


### 1.4. Execution via user interface

Use of Gradio library for web interface

<b>Note:</b> The environment variable ```AWS_API``` should point to the local container
> export AWS_API=http://localhost:8080

Command line for execution:
> python3 app.py

The Gradio web application should now be accessible at http://localhost:7860


## 2. Deployment to AWS

### 2.1. Pushing the docker container to AWS ECR

<details>

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

</details>

[Link to AWS ECR Documention](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html)

### 2.2. Creating and testing a Lambda function

<details>

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

</details>

### 2.3. Creating an API via API Gateway

<details>

<b>Steps</b>: 
 - Create a new ```Rest API``` (e.g. ```iris-classification-api```)
 - Add a new resource to the API (e.g. ```/classify```)
 - Add a ```POST``` method to the resource
 - Integrate the Lambda function to the API
   - Notes: using proxy integration option unchecked
 - Deploy API with a specific stage (e.g. ```test``` stage)

</details>

Example AWS API Endpoint:
```https://<api_id>.execute-api.<aws_region>.amazonaws.com/test/classify```


### 2.4. Execution for deployed model

Example of a prediction request

bash
> curl -X POST "https://<api_id>.execute-api.<aws_region>.amazonaws.com/test/classify" -H "Content-Type: application/json" -d '{"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}'

python
> python3 inference_api.py --url https://<api_id>.execute-api.<aws_region>.amazonaws.com/test/classify -d '{"features": [[6.5, 3.0, 5.8, 2.2], [6.1, 2.8, 4.7, 1.2]]}'


## 3. Deployment to Hugging Face

This web application is available on Hugging Face

Hugging Face space URL:
https://huggingface.co/spaces/cvachet/iris_classification_lambda

Note: This space uses the ML model deployed on AWS Lambda

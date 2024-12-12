# Dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY . ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Run test cases as this saves the ML model in the container
RUN pytest tests -s -v

CMD ["lambda.lambda_handler"]

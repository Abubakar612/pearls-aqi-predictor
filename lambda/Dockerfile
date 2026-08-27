FROM public.ecr.aws/lambda/python:3.13

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy Lambda application
COPY lambda/app.py ${LAMBDA_TASK_ROOT}/app.py

# Copy production models
COPY models/production ${LAMBDA_TASK_ROOT}/models/production

# Install Python dependencies
COPY lambda/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target ${LAMBDA_TASK_ROOT}

# Lambda handler
CMD ["app.lambda_handler"]
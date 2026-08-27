FROM public.ecr.aws/lambda/python:3.13

WORKDIR ${LAMBDA_TASK_ROOT}

COPY lambda/app.py ${LAMBDA_TASK_ROOT}/app.py
COPY lambda/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]
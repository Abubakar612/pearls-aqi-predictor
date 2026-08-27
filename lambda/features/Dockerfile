FROM public.ecr.aws/lambda/python:3.13

WORKDIR ${LAMBDA_TASK_ROOT}

COPY lambda/features/app.py ${LAMBDA_TASK_ROOT}/app.py
COPY models/production/feature_list.json ${LAMBDA_TASK_ROOT}/feature_list.json
COPY lambda/features/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target ${LAMBDA_TASK_ROOT}

CMD ["app.lambda_handler"]

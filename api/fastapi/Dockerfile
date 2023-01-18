FROM tiangolo/uvicorn-gunicorn:python3.10

RUN mkdir /fastapi
RUN mkdir /fastapi/models
RUN mkdir /fastapi/models/checkpoints

COPY api/fastapi/requirements_api.txt /fastapi
COPY requirements.txt /fastapi
RUN pip install -r /fastapi/requirements_api.txt
RUN pip install -r /fastapi/requirements.txt

COPY models/model.py /fastapi/models
COPY api/fastapi/. /fastapi
COPY models/checkpoints/epoch=11-step=6624.ckpt /fastapi/models/checkpoints
COPY src/ /fastapi/src/

WORKDIR /fastapi

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
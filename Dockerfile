FROM tiangolo/uvicorn-gunicorn:python3.9-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV MODULE_NAME=app
ADD requirements.txt .
ENV PYHTONUNBUFFERED=1
RUN apt-get update \
  && apt-get -y install tesseract-ocr
RUN apt-get install poppler-utils -y
RUN pip install -r requirements.txt && rm -rf /root/.cache
COPY . .
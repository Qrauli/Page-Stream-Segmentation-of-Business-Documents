FROM python:3.9

WORKDIR /app
#ENV DEBIAN_FRONTEND=noninteractive
#ENV MODULE_NAME=app
ADD requirements.txt .
#ENV PYHTONUNBUFFERED=1
#ENV BATCH_SIZE=30
RUN apt-get update \
  && apt-get -y install tesseract-ocr
RUN apt-get install poppler-utils -y
RUN pip install -r requirements.txt && rm -rf /root/.cache
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

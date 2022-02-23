from tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY src /app/src
WORKDIR /app/src
ENTRYPOINT ["python", "vectorise_csv.py"]
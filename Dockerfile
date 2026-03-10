FROM python:3.12-slim

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
COPY app.py /app/

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]

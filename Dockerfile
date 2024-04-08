FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && rm -rf /var/lib/apt/list/*
RUN pip3 install -r requirements.txt

COPY *.py /app/
COPY config.json /app/
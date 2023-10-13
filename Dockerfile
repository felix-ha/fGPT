FROM python:3.11.6-bookworm

RUN apt update && apt install -y git vim curl wget 

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements-dev.txt

COPY . . 

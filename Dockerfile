FROM python:3.11.6-bookworm

RUN apt update && apt install -y git vim curl wget 

WORKDIR /app
COPY requirements-dev.txt requirements-dev.txt ./
COPY requirements-data.txt requirements-data.txt ./
COPY requirements.txt requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . . 

RUN wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -P /app/data
RUN wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt -P /app/data

ENTRYPOINT [ "python", "main.py" , "--full", "--ratio=0.0001", "--splits=1000", "--epochs=1"]

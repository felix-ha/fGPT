FROM python:3.11.6-bookworm

RUN apt update && apt install -y git vim curl wget 

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements-dev.txt

RUN wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -P /app/data
RUN wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt -P /app/data

COPY . . 

ENTRYPOINT [ "python", "main.py" ]

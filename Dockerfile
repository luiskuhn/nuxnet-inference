FROM python:3.12-slim-bookworm

RUN apt update && apt-get install -y procps wget && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel

CMD ["nuxnet-pred"]

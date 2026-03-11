FROM python:3.12-slim-bookworm

RUN apt update && apt-get install -y procps wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install .

CMD ["nuxnet-pred"]

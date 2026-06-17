FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV FAISS_PATH=/app/data/faiss_index

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN python3 scripts/export_models.py

RUN mkdir -p /app/data /app/models

EXPOSE 9103

CMD ["python3", "main.py"]

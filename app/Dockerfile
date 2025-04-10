FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app
COPY . /app/

RUN pip install --upgrade pip && \
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir -r requirements.txt
    
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
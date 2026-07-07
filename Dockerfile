FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create output directory
RUN mkdir -p /data/generated

WORKDIR /opt/kortex

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY worker/ ./worker/

EXPOSE 8080

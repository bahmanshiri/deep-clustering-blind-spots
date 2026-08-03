FROM python:3.11-slim

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Node.js is only needed to rebuild the paper docx from generate_paper.js
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
    && npm install -g docx \
    && rm -rf /var/lib/apt/lists/*

COPY code/ code/
COPY data/ data/
COPY figures/ figures/

WORKDIR /work/code
CMD ["python3", "generate_hard_dataset.py"]

FROM python:3.11-slim

WORKDIR /app

# System deps for RDKit + LightGBM
RUN apt-get update && apt-get install -y \
    libxrender1 libxext6 libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs models

EXPOSE 7860

CMD ["python", "app.py"]

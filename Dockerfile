# voicematch-services/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Instala dependências de sistema para decodificação de áudio (FFmpeg e libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Mesma porta usada localmente (PORT=8001 no .env)
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

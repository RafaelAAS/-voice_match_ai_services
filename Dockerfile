# voicematch-services/Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Instala dependências primeiro (aproveita cache do Docker se requirements.txt não mudar)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Mesma porta usada localmente (PORT=8001 no .env)
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

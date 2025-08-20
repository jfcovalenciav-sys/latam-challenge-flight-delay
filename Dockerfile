# Runtime compatible con pandas 1.3.x / numpy 1.22.x / sklearn 1.3.0
FROM python:3.10-slim

# Buenas prácticas en Python + pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Dependencia de runtime para OpenMP usada por scikit-learn/scipy
# (sin esto en slim, a veces faltan libs en tiempo de ejecución)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos requisitos primero para aprovechar la cache de capas
COPY requirements.txt constraints.txt ./

# Importante: instalar con constraints para fijar starlette/anyio/httpx/scipy
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copiamos el resto del proyecto (incluye data/ si entrenas en startup)
COPY . .

EXPOSE 8080

# Cloud Run/containers: toma el puerto de $PORT
# Usa forma "shell" para que se expanda la variable de entorno
CMD ["sh","-c","uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]

# 1. Usar Linux con Python 3.12 como base (versión slim para que sea muy ligero)
FROM python:3.12-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de requerimientos PRIMERO
# (Esto es un truco Senior: Docker guarda esto en caché. Si cambias tu código pero no tus dependencias, esto no se vuelve a descargar)
COPY requirements.txt .

# 4. Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copiar tu código de la API y los modelos entrenados
COPY src/api.py src/api.py
COPY models/ models/

# 6. Exponer el puerto por el que escuchará la API
EXPOSE 8000

# 7. El comando exacto que ejecutará el contenedor al encenderse
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
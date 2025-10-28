# Utiliser une image officielle de Python
FROM python:3.11-slim

# Empêche Python de créer des fichiers .pyc et garde les logs visibles
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Créer le dossier de travail
WORKDIR /app

# Installer quelques dépendances système de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt et installer les libs Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier tout le projet dans le conteneur
COPY . .

# Déclarer le port utilisé (Render/Railway lisent cette info)
EXPOSE 8000

# Démarrer ton application avec le script start.sh
CMD ["bash", "start.sh"]

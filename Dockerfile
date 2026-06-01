# 1. Image de base Python
FROM python:3.10-slim

# 2. Dossier de travail
WORKDIR /app

# 3. Installation des outils système essentiels
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. On copie d'abord le fichier requirements pour optimiser le cache Docker
COPY requirements.txt ./

# 5. Installation des dépendances Python via le fichier requirements
RUN pip install --no-cache-dir -r requirements.txt

# 6. Téléchargement amont des données NLTK nécessaires pour le preprocessing
RUN python -m nltk.downloader wordnet

# 7. Copie du reste des fichiers de l'application (code, modèles et dataset)
COPY app.py darija_sentiment_dataset.csv ./
COPY models/ ./models/

# 8. Exposition du port Streamlit
EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 9. Lancement du serveur Streamlit
CMD ["streamlit", "run", "app.py"]
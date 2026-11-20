FROM python:3.12-slim
WORKDIR /srv
ENV FASTEMBED_CACHE_PATH=/srv/models PYTHONUNBUFFERED=1 PORT=7860
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# bake embedding models into the image so cold starts are fast and offline-safe
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); SparseTextEmbedding('Qdrant/bm25')"
COPY app app
COPY data data
COPY eval eval
EXPOSE 7860
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

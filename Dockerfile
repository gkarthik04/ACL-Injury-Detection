# Container for deploying the Flask inference API (e.g. Hugging Face Spaces, Docker SDK)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY backend/ backend/
COPY docs/ docs/
# Place the trained checkpoint at backend/best_model.pth (or set MODEL_PATH)
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn -w 1 -t 180 -b 0.0.0.0:${PORT} --chdir backend app:app"]

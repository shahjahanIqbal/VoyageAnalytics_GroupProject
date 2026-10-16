FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies the existing app, model files, datasets, and scripts that are
# present in the repository and not excluded by .dockerignore.
# Note: model .joblib files are generated locally (see README.md setup
# order) and are copied here when present in the build context.
COPY . .

EXPOSE 5001 8501

RUN chmod +x docker-start.sh

# Starts the Flask recommendation API (port 5001) and the Streamlit
# interface (port 8501) in the same container.
CMD ["./docker-start.sh"]

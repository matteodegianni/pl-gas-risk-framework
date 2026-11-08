# Daily risk pipeline image (Stage 16).
# Build:  docker build -t gasrisk .
# Run:    docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" -v "$PWD/exports:/app/exports" gasrisk
# Licensed data stay outside the image: mount data/ (with the Investing.com exports in data/raw/investing/).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Warsaw

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN sed -i '/^-e \./d' requirements.txt \
    && pip install -r requirements.txt \
    && pip install --no-deps .
COPY config ./config
COPY handover ./handover
COPY outputs ./outputs

RUN useradd --create-home risk && chown -R risk /app
USER risk

ENTRYPOINT ["python", "-W", "ignore", "-m", "gasrisk.pipeline"]
CMD []

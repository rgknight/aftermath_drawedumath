FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . review_app/
# Source files may arrive owner-only (0600); the app runs as a non-root user, so open read access.
RUN chmod -R a+rX /srv/review_app && useradd --create-home reviewer
USER reviewer

CMD exec uvicorn review_app.app:create_app_from_env --factory \
    --host 0.0.0.0 --port "${PORT:-8080}" --proxy-headers --forwarded-allow-ips "*"

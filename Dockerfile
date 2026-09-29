# Ocean Police - single web service: built React app + FastAPI on one port.
# Used by Render (render.yaml). Scene processing is NOT done here (see .github/workflows/live-update.yml).

FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
# The lockfile was generated on Windows; npm can omit Linux-only optional binaries
# (rollup/esbuild) from it, so fall back to a fresh install if `npm ci` breaks.
RUN npm ci --no-audit --no-fund && node -e "require('rollup')"     || (rm -rf node_modules package-lock.json && npm install --no-audit --no-fund)
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY backend ./backend
COPY models/model_card.json ./models/model_card.json
COPY --from=web /web/dist ./frontend/dist
ENV PYTHONPATH=/app/src \
    OILSPILL_API_PROCESSING=0 \
    OILSPILL_DATA_DIR=/tmp/oilspill \
    OILSPILL_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]

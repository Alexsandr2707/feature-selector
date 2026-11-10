FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install --no-cache-dir uv==0.13.4
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY services/api-requirements.txt ./services/api-requirements.txt
COPY services/ml/requirements.txt ./services/ml/requirements.txt
RUN uv pip install --python /opt/venv/bin/python -r services/ml/requirements.txt
COPY services ./services
COPY configs ./configs
COPY deploy ./deploy
ENV PATH="/opt/venv/bin:$PATH"
CMD ["python", "-m", "services.launch", "ml"]

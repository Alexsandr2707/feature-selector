FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
COPY services/api-requirements.txt ./services/api-requirements.txt
COPY services/master/requirements.txt ./services/master/requirements.txt
RUN pip install --no-cache-dir -r services/master/requirements.txt
COPY services ./services
COPY deploy ./deploy
CMD ["python", "-m", "services.launch", "master"]

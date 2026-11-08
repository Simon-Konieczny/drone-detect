# Inference image: ONNX Runtime only, no PyTorch, no training code, no data.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-infer.txt .
RUN pip install -r requirements-infer.txt

COPY src/predict.py .
COPY models/model.onnx models/model.onnx

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

ENTRYPOINT ["python", "predict.py", "--model", "models/model.onnx"]
CMD ["--help"]
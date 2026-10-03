FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
# The dashboard reads only precomputed artifacts, so heavy model libraries are not needed.
# Run the pipeline locally before building: make run
COPY artifacts ./artifacts
RUN pip install --no-cache-dir ".[app]"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

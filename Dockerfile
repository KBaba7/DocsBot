FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user app ./app

RUN pip install --no-cache-dir -e .

# Uploads directory writable by non-root user
RUN mkdir -p /home/user/app/uploads && chown -R user /home/user/app/uploads

USER user

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY reproducibility/requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement /tmp/requirements.lock

COPY Makefile ./Makefile
COPY reproducibility ./reproducibility

CMD ["sh", "-c", "make reproduce && make verify"]

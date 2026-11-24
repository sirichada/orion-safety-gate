# Orion Safety Gate, packaged for a local NVIDIA GPU.
# The image holds the code and its Python dependencies only. Model weights (Hugging Face cache), data/,
# runs/ and the API key are supplied at run time by docker-compose.yml, never baked in.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf \
    PATH=/opt/venv/bin:$PATH

# build-essential and python3-dev: fasttext has no wheel and is built from source.
# curl and bash: scripts/fetch_data.sh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

# Exact versions of the packages the suite was validated with; requirements.txt lists names only.
COPY requirements.txt constraints-docker.txt ./

# torch comes from the CUDA 12.8 index (the RTX 50 series needs it); everything else from PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 -c constraints-docker.txt torch \
    && pip install -c constraints-docker.txt "setuptools<81" pybind11 \
    && pip install --no-build-isolation -c constraints-docker.txt -r requirements.txt

COPY scripts ./scripts
COPY config ./config
COPY templates ./templates

ENTRYPOINT ["python", "scripts/safety_suite.py"]
CMD ["--help"]

# syntax=docker/dockerfile:1

# Admetus: ADMET property prediction from SMILES.
#
#   docker run --rm -v "$PWD:/data" ghcr.io/ytworks/admetus endpoints
#   docker run --rm -v "$PWD:/data" ghcr.io/ytworks/admetus \
#       predict -i /data/in.csv -o /data/out.csv --endpoint dili
#
# Two stages so the wheel-building toolchain does not ship. The runtime stage
# installs libgomp1 because LightGBM's wheels link against an OpenMP runtime they
# do not bundle, and the import fails with a bare dlopen error without it.

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install '.[train]'


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Admetus" \
      org.opencontainers.image.description="ADMET property prediction from SMILES with LightGBM fold ensembles" \
      org.opencontainers.image.source="https://github.com/ytworks/Admetus" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ADMETUS_MODELS_DIR=/models \
    ADMETUS_DATA_DIR=/data/datasets \
    ADMETUS_CACHE_DIR=/tmp/admetus-cache

# Run as a non-root user, with the mount point owned so a bind-mounted working
# directory is writable without extra flags.
RUN useradd --create-home --uid 10001 admetus \
 && mkdir -p /data /models \
 && chown -R admetus:admetus /data /models

USER admetus
WORKDIR /data
VOLUME ["/data", "/models"]

ENTRYPOINT ["admetus"]
CMD ["--help"]

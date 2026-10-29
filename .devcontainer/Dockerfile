FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/llm-studies/.venv \
    PATH="/llm-studies/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential ninja-build \
    python3 python3-dev python3-venv \
    openssh-client sudo less vim htop tmux \
    && rm -rf /var/lib/apt/lists/*

# UV and Dependencies
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/

# No-Root User Setup
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

USER ${USERNAME}

# Created as dev so the named volumes mounted here are initialised with the right owner
RUN mkdir -p ~/.claude ~/.cache/huggingface ~/.cache/uv

WORKDIR /llm-studies
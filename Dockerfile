FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# System dependencies for building and running vision/video stacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda.
ENV CONDA_DIR=/opt/conda
RUN set -eux; \
    ARCH="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${ARCH}" in \
      amd64|x86_64) CONDA_ARCH="x86_64" ;; \
      arm64|aarch64) CONDA_ARCH="aarch64" ;; \
      *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac; \
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${CONDA_ARCH}.sh" -O /tmp/miniconda.sh; \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR}; \
    rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace/SFA

# Copy dependency list first for better layer caching.
COPY requirements.txt /workspace/SFA/requirements.txt

# Equivalent to:
# conda create -n sfa python=3.10
# conda activate sfa
# pip install -r requirements.txt
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n sfa python=3.10 -y && \
    conda run -n sfa python -m pip install --upgrade pip setuptools wheel && \
    conda run -n sfa python -m pip install --no-cache-dir \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    grep -Ev '^(torch|torchvision|torchaudio)==.*' requirements.txt > /tmp/requirements.no_torch.txt && \
    conda run -n sfa python -m pip install --no-cache-dir -r /tmp/requirements.no_torch.txt

# Copy full project and build third-party extensions used by GoMatching.
COPY . /workspace/SFA
RUN conda run -n sfa bash -lc ' \
    export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" C_INCLUDE_PATH=/usr/local/cuda/include CPLUS_INCLUDE_PATH=/usr/local/cuda/include; \
    cd GoMatching/third_party; \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    python -m pip install --no-cache-dir termcolor yacs tabulate cloudpickle matplotlib tensorboard rapidfuzz Polygon3 shapely scikit-image numba; \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps -e . \
'

# Make sfa env default in interactive shells.
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate sfa" >> /root/.bashrc

ENV PATH=${CONDA_DIR}/envs/sfa/bin:${CONDA_DIR}/bin:${PATH}

CMD ["bash"]

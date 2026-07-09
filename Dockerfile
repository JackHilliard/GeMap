# GeMap (ECCV'24) — https://github.com/cnzzx/GeMap
# Follows docs/install.md: conda py3.8, torch 1.9.1+cu111, mmcv-full 1.4.0,
# mmdet 2.14.0, mmsegmentation 0.14.1, mmdetection3d (vendored) + GKT CUDA op.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Architectures to build the custom CUDA ops for (Pascal -> Ampere).
# No GPU is visible at `docker build` time, so this must be listed explicitly
# rather than auto-detected.
ENV FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/conda/envs/gemap/bin:/opt/conda/bin:${PATH}"

# Drop NVIDIA's apt repo: its signing key baked into this base image is
# expired/rotated, and we don't need it (CUDA toolkit is already installed).
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ninja-build \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniconda + a "gemap" env, matching docs/install.md's
# `conda create -n gemap python=3.8`. The env's bin/ is prepended to PATH
# above so plain `python`/`pip` in later RUNs resolve to it directly,
# without needing `conda activate` (which requires shell hooks in RUN).
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /opt/conda/bin/conda create -y -n gemap python=3.8 \
    && /opt/conda/bin/conda clean -afy

RUN pip install --no-cache-dir \
        torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
        -f https://download.pytorch.org/whl/torch_stable.html

# mmcv-full prebuilt wheel for cu111/torch1.9.0 (compatible with torch 1.9.1)
RUN pip install --no-cache-dir mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    && pip install --no-cache-dir mmdet==2.14.0 mmsegmentation==0.14.1 timm==0.4.12

WORKDIR /workspace/GeMap
COPY . .

# Pre-install mmdetection3d's runtime deps via pip (wheels). Without this,
# `setup.py develop` falls back to setuptools' legacy easy_install to fetch
# unpinned deps (e.g. scikit-image), which no longer ships sdists it can
# build and fails.
RUN pip install --no-cache-dir -r mmdetection3d/requirements/runtime.txt

# Build the vendored mmdetection3d
RUN cd mmdetection3d && python setup.py develop

# Build GeMap's custom Geometric Kernel Attention (GKT) CUDA op
RUN cd projects/mmdet3d_plugin/gemap/modules/ops/geometric_kernel_attn \
    && python setup.py build install

# Remaining Python deps (shapely, av2 — av2 needs Python >=3.8)
RUN pip install --no-cache-dir -r requirement.txt

# mmdet3d's runtime.txt pins numpy<1.20 + numba==0.48.0, but av2/Pillow
# need numpy>=1.21 (numpy.typing.NDArray). numba==0.48.0 only supports
# numpy<=1.18, so bump both together to a pair that's mutually compatible.
RUN pip install --no-cache-dir "numpy==1.22.4" "numba==0.56.4"

RUN mkdir -p ckpts data

# Datasets (./data) and checkpoints (./ckpts) are expected to be mounted at
# runtime, e.g.:
#   docker run --gpus all -it \
#     -v /path/to/data:/workspace/GeMap/data \
#     -v /path/to/ckpts:/workspace/GeMap/ckpts \
#     gemap:latest
CMD ["/bin/bash"]

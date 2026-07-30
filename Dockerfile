# GeMap (ECCV'24) — https://github.com/cnzzx/GeMap
# Diverges from docs/install.md (conda py3.8, torch 1.9.1+cu111, mmcv-full
# 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1): that stack predates CUDA 11.8,
# the first CUDA release with real Hopper (H100, sm_90) support. Bumped to
# torch 2.1.0/CUDA 11.8 + matching mmcv/mmdet/mmsegmentation, following the
# same fix applied to the sibling MapTRv2 codebase's Dockerfile.
ARG PYTORCH="2.1.0"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 8.6 covers Ampere (e.g. RTX 3070/3090); 9.0+PTX covers Hopper (H100) and,
# via PTX forward-compat JIT, anything newer released after this image (e.g.
# Blackwell RTX 50-series).
ENV TORCH_CUDA_ARCH_LIST="8.6 9.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ninja-build \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation. Bumped from the versions in
# docs/install.md (mmcv-full==1.4.0/mmdet==2.14.0/mmsegmentation==0.14.1,
# built for CUDA 11.1) to support Hopper (H100) GPUs, which need CUDA>=11.8 --
# the first CUDA release with real sm_90 support. mmdet/mmsegmentation are
# bumped alongside mmcv because both hard-assert an mmcv upper bound in their
# own __init__.py that the old pins don't cover (mmdet<=1.4.0,
# mmsegmentation<=1.4.0); 2.28.2/0.30.0 are the last mmdet 2.x/mmsegmentation
# 0.x releases, with mmcv upper bounds of 1.8.0, comfortably covering 1.7.2.
#
# mmcv-full MUST be built from source here, not installed from OpenMMLab's
# prebuilt wheel index. Their prebuilt cu118/torch2.1.0 wheel was compiled
# with OpenMMLab's own (unknown, non-configurable) TORCH_CUDA_ARCH_LIST,
# which does not include Hopper (sm_90) and has no +PTX fallback baked in --
# this is invisible on Ampere and only surfaces on an actual H100 as
# `RuntimeError: CUDA error: no kernel image is available for execution on
# the device` inside mmcv's own CUDA ops (e.g. ms_deform_attn,
# sigmoid_focal_loss) the first time they run. Building from source makes
# mmcv's setup.py pick up this Dockerfile's own TORCH_CUDA_ARCH_LIST (set
# above, "8.6 9.0+PTX"), same as every other custom op in this image (GKT,
# mmdetection3d's ops) already does.
RUN MMCV_WITH_OPS=1 pip install --no-cache-dir --no-binary mmcv-full "mmcv-full==1.7.2"
# mmcv 1.7.2's single-GPU MMDataParallel path (mmcv.parallel.Scatter.forward,
# used by tools/gemap/benchmark.py, tools/gemap/vis_pred.py and the
# bevformer train/test apis) calls PyTorch's private
# torch.nn.parallel._functions._get_stream() with a raw int device id;
# torch>=2.x's version of that function requires a torch.device object
# instead (`AttributeError: 'int' object has no attribute 'type'`). Patch
# the installed file to wrap the device id.
RUN sed -i \
    "s/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) else device) for device in target_gpus]/" \
    /opt/conda/lib/python3.10/site-packages/mmcv/parallel/_functions.py
RUN pip install mmdet==2.28.2
RUN pip install mmsegmentation==0.30.0
# Unpinned `timm` pulls in the latest release, which needs torch.fx APIs not
# present in older torch; pin to a version contemporaneous with this codebase.
RUN pip install timm==0.6.13

# spconv2 (maintained fork) replaces mmdetection3d's vendored spconv 1.x,
# whose CUDA kernels are incompatible with CUDA 11.8/sm_86+90 (crash with
# "cuda execution failed with error 2" even on trivial inputs) -- see
# mmdetection3d/mmdet3d/ops/spconv/__init__.py, which re-exports this
# instead of the vendored implementation.
RUN pip install --no-cache-dir spconv-cu118==2.3.8

RUN conda clean --all

WORKDIR /workspace/GeMap
COPY . .

# Pre-install mmdetection3d's runtime deps via pip (wheels). Without this,
# `setup.py develop` falls back to setuptools' legacy easy_install to fetch
# unpinned deps (e.g. scikit-image), which no longer ships sdists it can
# build and fails.
RUN pip install --no-cache-dir -r mmdetection3d/requirements/runtime.txt

# Build the vendored mmdetection3d (v0.17.2, patched for torch 2.1: THC/THC.h
# was removed from ATen, and its vendored spconv 1.x ops now re-export
# spconv2 -- see mmdetection3d/mmdet3d/ops/spconv/__init__.py).
RUN cd mmdetection3d && python setup.py develop

# Build GeMap's custom Geometric Kernel Attention (GKT) CUDA op (also patched
# for torch 2.1's ATen header/C++ standard requirements).
RUN cd projects/mmdet3d_plugin/gemap/modules/ops/geometric_kernel_attn \
    && python setup.py build install

# Remaining Python deps (shapely, av2 -- av2 needs Python >=3.8)
RUN pip install --no-cache-dir -r requirement.txt

# av2 imports numpy.typing (added in numpy 1.20) at module load time. Pin
# below 1.24 to keep the np.float/np.int aliases mmdet/mmcv-full may still
# use (removed in numpy 1.24).
RUN pip install --no-cache-dir "numpy==1.23.5"

# mmcv-full/mmdet/nuscenes-devkit/av2 all transitively pull in non-headless
# opencv-python, which links its GUI backend against libGL.so.1. Swap to the
# headless build -- same OpenCV, no GL/X11 linkage at all -- so cv2 never
# needs a system libGL.so.1 (e.g. on Singularity/Apptainer clusters with
# --nv where the host's newer libGL.so.1 can get bind-mounted over the
# container's own and fail to load against this image's glibc).
RUN OPENCV_VERSION=$(pip show opencv-python | sed -n 's/^Version: //p') \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless==${OPENCV_VERSION}"

RUN mkdir -p ckpts data

# Datasets (./data) and checkpoints (./ckpts) are expected to be mounted at
# runtime, e.g.:
#   docker run --gpus all -it \
#     -v /path/to/data:/workspace/GeMap/data \
#     -v /path/to/ckpts:/workspace/GeMap/ckpts \
#     gemap:latest
CMD ["/bin/bash"]

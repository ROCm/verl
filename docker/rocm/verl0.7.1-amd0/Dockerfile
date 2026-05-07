FROM ubuntu:22.04 AS ubuntu

#
# Install basic packages from OS distro
#
FROM ubuntu AS base

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && \
    apt install -y git software-properties-common curl rsync dialog gfortran wget sqlite3

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    if ! python3 --version | grep -q ${PYTHON_VERSION} ; then \
    add-apt-repository -y ppa:deadsnakes/ppa && apt update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
                    python${PYTHON_VERSION}-lib2to3 python-is-python3 ; fi

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}



RUN wget -nv -O /tmp/cmake-3.26.4-linux-x86_64.tar.gz https://cmake.org/files/v3.26/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar zfx /tmp/cmake-3.26.4-linux-x86_64.tar.gz -C /opt/ && \
    mv /opt/cmake-3.26.4-linux-x86_64 /opt/cmake-3.26.4 && \
    rm -f /tmp/cmake-3.26.4-linux-x86_64.tar.gz

ENV PATH=/opt/cmake-3.26.4/bin:$PATH

#
# Install ROCm rpm packages
#
FROM base AS rocm_deb

ARG ROCM_VERSION=7.0.2
ARG AMDGPU_VERSION=7.0.2

RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main\n" | tee /etc/apt/sources.list.d/rocm.list \
    && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main\n" | tee /etc/apt/sources.list.d/amdgpu.list \
    && printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" | tee /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y rocm && \
    find /opt/rocm/lib -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/hipblaslt/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/rocblas/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/share/miopen/db -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    sqlite3 /opt/rocm/lib/rocfft/rocfft_kernel_cache.db "delete from cache_v1 where arch != '${GFX_ARCH}' ; vacuum"

ENV ROCM_HOME=/opt/rocm
ENV CPLUS_INCLUDE_PATH=/opt/rocm/include:
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

#
# Install pytorch packages
#
FROM rocm_deb AS rocm_torch

ARG GFX_ARCH=gfx942

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip "setuptools<80" wheel numpy einops ninja && \
    pip install /opt/rocm/share/amd_smi

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torch-2.9.1.dev20251204+rocm7.0.2.lw.git351ff442-cp312-cp312-linux_x86_64.whl -O torch-2.9.1.dev20251204+rocm7.0.2.lw.git351ff442-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/apex-1.9.0a0+rocm7.0.2.git07c3ee53-cp312-cp312-linux_x86_64.whl -O apex-1.9.0a0+rocm7.0.2.git07c3ee53-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchaudio-2.9.0+rocm7.0.2.gite3c6ee2b-cp312-cp312-linux_x86_64.whl -O torchaudio-2.9.0+rocm7.0.2.gite3c6ee2b-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchvision-0.24.0+rocm7.0.2.gitb919bd0c-cp312-cp312-linux_x86_64.whl -O torchvision-0.24.0+rocm7.0.2.gitb919bd0c-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/triton-3.5.1+rocm7.0.2.gita272dfa8-cp312-cp312-linux_x86_64.whl -O triton-3.5.1+rocm7.0.2.gita272dfa8-cp312-cp312-linux_x86_64.whl && \
    pip3 install *.whl && \
    rm -f *.whl

ENV PYTORCH_ROCM_ARCH=${GFX_ARCH}

WORKDIR /root

FROM rocm_torch AS new_toolset

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy einops packaging psutil ninja

#
# Build Flash Attention wheel
#
FROM new_toolset AS fa_build

ARG FA_REPO="https://github.com/ROCm/flash-attention"
ARG FA_TAG="8afc617aa7ebb72f70f659b03b9908c30920ad37"

RUN git clone ${FA_REPO} \
&& cd flash-attention \
&& git checkout ${FA_TAG} \
&& git submodule init \
&& git submodule update

RUN cd flash-attention \
&& GPU_ARCHS=gfx942 BUILD_TARGET=rocm MAX_JOBS=$(nproc) python3 setup.py bdist_wheel \
&& mkdir /install && cp dist/*.whl /install

#
# Install Flash Attention
#
FROM new_toolset AS install_fa

RUN --mount=type=bind,from=fa_build,source=/install,target=/tmp/install \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/install/*.whl

#
# Install TE
#
FROM install_fa AS te

ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_USE_ROCM=1
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=gfx942
ENV NVTE_USE_CAST_TRANSPOSE_TRITON=0
ENV NVTE_CK_USES_BWD_V3=1
ENV NVTE_CK_V3_BF16_CVT=2

ARG TE_TAG="307b5e86bb5960e347f1f70224d17e09e73f5338"
RUN pip install pybind11 && git clone https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && git checkout $(TE_TAG) && git submodule update --init --recursive && \
    GPU_ARCHS=gfx942 MAX_JOBS=$(nproc) python3 setup.py install && \
    cd .. && rm -rf TransformerEngine
#
# Install vllm
#
FROM te AS install_vllm

# waiting for upstream vllm update the patch
ENV PYTORCH_ROCM_ARCH="gfx942"
ARG VLLM_TAG="71161e8b63f9534d6ac5e098a4874621164d1f1e"
RUN mkdir /workspace
COPY 71161e8b6_patch.patch /workspace/71161e8b6_patch.patch
 
RUN cd /workspace && pip install setuptools_scm && \
    ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so && \
    git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && git checkout $(VLLM_TAG) && \
    mv /workspace/71161e8b6_patch.patch . && git apply 71161e8b6_patch.patch \
    pip install -r requirements/rocm.txt && \
    MAX_JOBS=32 python3 setup.py develop --no-deps

FROM install_vllm AS install_verl
ARG VERL_TAG="e066848231f7ae3f32255e5e555ea239b9387890"
RUN pip install cupy-rocm-7-0
RUN cd /workspace && git clone -b amd-integration https://github.com/ROCm/verl.git && \
    cd verl && git checkout $(VERL_TAG) && pip install -e .


ENV MIOPEN_DEBUG_CONV_DIRECT=0
ARG AITER_TAG="f02877b8c29a9c3469af5ee0a400fffc14b805c9"
RUN apt install vim -y
RUN cd /workspace && git clone --recursive https://github.com/ROCm/aiter.git && cd aiter \
    git checkout $(AITER_TAG) && python3 setup.py develop

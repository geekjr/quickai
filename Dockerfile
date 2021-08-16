# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.7    (apt)
# torch         latest (git)
# pytorch       latest (pip)
# tensorflow    latest (pip)
# ==================================================================

FROM ubuntu:18.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

    # ==================================================================
    # tools
    # ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    build-essential \
    apt-utils \
    ca-certificates \
    wget \
    git \
    vim \
    libssl-dev \
    curl \
    unzip \
    unrar \
    && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

    # ==================================================================
    # python
    # ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    python3.7 \
    python3.7-dev \
    python3-distutils-extra \
    && \
    wget -O ~/get-pip.py \
    https://bootstrap.pypa.io/get-pip.py && \
    python3.7 ~/get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    $PIP_INSTALL \
    setuptools \
    && \
    $PIP_INSTALL \
    numpy \
    scipy \
    pandas \
    cloudpickle \
    scikit-image>=0.14.2 \
    scikit-learn \
    matplotlib \
    Cython \
    tqdm \
    && \

    # ==================================================================
    # torch
    # ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    sudo \
    && \

    $GIT_CLONE https://github.com/nagadomi/distro.git ~/torch --recursive && \
    cd ~/torch && \
    bash install-deps && \
    sed -i 's/${THIS_DIR}\/install/\/usr\/local/g' ./install.sh && \
    ./install.sh && \

    # ==================================================================
    # pytorch
    # ------------------------------------------------------------------

    $PIP_INSTALL \
    future \
    numpy \
    protobuf \
    enum34 \
    pyyaml \
    typing \
    && \
    $PIP_INSTALL \
    --pre torch torchvision -f \
    https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html \
    && \

    # ==================================================================
    # tensorflow
    # ------------------------------------------------------------------

    $PIP_INSTALL \
    tensorflow \
    && \

    # ==================================================================
    # other
    # ------------------------------------------------------------------

    $PIP_INSTALL \
    transformers \
    opencv-python \
    quickai \
    && \

    # ==================================================================
    # config & cleanup
    # ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006

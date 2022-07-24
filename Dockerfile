FROM tensorflow/tensorflow:2.5.0-gpu

# Addressing GPG error
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install OpenEXR
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
	libopenexr-dev && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/jamesbowman/openexrpython.git
RUN pip3 install openEXR==1.3.0
WORKDIR /openexrpython
RUN python3 setup.py install

# Run self-supervised post-correction
WORKDIR /codes
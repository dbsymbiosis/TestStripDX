FROM nvidia/cuda:10.2-base-ubuntu18.04

LABEL maintainer="Timothy Stephens"

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-client libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

# Install mamba
RUN conda install -y mamba -c conda-forge

# Setup git
RUN mkdir -p  /home/$USERNAME/.ssh/ && \
    echo -e "Host github.com\n\tStrictHostKeyChecking no\n" >> /home/$USERNAME/.ssh/config

# Download and install TestStripDX
RUN cd /home/$USERNAME/ && \
    git clone https://github.com/dbsymbiosis/TestStripDX.git && \
    chown -R "$USERNAME:users" TestStripDX && \
    cd TestStripDX/ && \
    git checkout dev-StaticCoords
ENV PATH $PWD:$PATH
WORKDIR /home/$USERNAME/TestStripDX

#RUN git checkout dev
RUN mamba env update --file conda.yml && \
    conda clean -tipy

# For interactive shell
RUN conda init bash && \
    echo "conda activate TestStripDX" >> /home/$USERNAME/.bashrc && \
    echo 'export PATH=/home/$USER/TestStripDX:$PATH' >> /home/$USERNAME/.bashrc

# Compile model (copy files first)
COPY --chown="$USERNAME:users" ../models/* /home/$USERNAME/TestStripDX/models/
RUN conda run -n TestStripDX ./TestStripDX.py convert

# /home/docker/TestStripDX/TestStripDX.py
ENTRYPOINT [ "conda", "run", "-n", "TestStripDX", "TestStripDX.py" ]


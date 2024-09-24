FROM dolfinx/dolfinx:stable

ARG PYVISTA_VERSION=0.43.2
ARG PYTHON_VERSION=3.10


# Dependencies for pyvista and related packages

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig

RUN echo ${TARGETPLATFORM}

RUN echo "PLATFORM" ${TARGETPLATFORM}

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.2.6-cp310/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl"; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then python3 -m pip install vtk; fi

RUN dpkgArch="$(dpkg --print-architecture)"; \
    case "$dpkgArch" in arm64) \
    python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.2.6-cp310/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl" ;; \
    esac;
RUN python3 -m pip install pyvista


RUN pip3 install matplotlib ipython pandas
RUN pip3 install black sympy flake8 autoflake8
RUN pip3 install tqdm colorcet

RUN apt-get update --assume-yes 

# Install libosmesa
RUN apt-get install libosmesa6-dev  --assume-yes 






# Use the specified PyTorch ROCm base image
FROM rocm/pytorch:rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.4.0

# Set working directory
WORKDIR /app

# Environment variables
ARG GPU_ARCHS=gfx942
ARG MAX_JOBS=$(nproc)

# Install Dao AILab official fork of flash-attention
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git@75f90d60f348af768625b6ab6ce13e800c5bc48a -v

# Copy local Instella files
RUN mkdir -p /app/Instella
COPY . /app/Instella/

# Now install Instella
WORKDIR /app/Instella
RUN pip install -e .[all]

# Set the default command when the container starts
CMD ["/bin/bash"]

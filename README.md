# cuVS Benchmark

- [Introduction](#introduction)
- [Quickstart](#quickstart)
- [Setting up a new host](#setting-up-a-new-host)
  - [Update CUDA toolkit](#update-cuda-toolkit)
  - [Install docker](#install-docker)
  - [Install NVIDIA Container Toolkit](#install-nvidia-container-toolkit)
  - [Install mamba (conda)](#install-mamba-conda)
- [Deprecated](#deprecated)
  - [Install the `raft-ann-bench` tool](#install-the-raft-ann-bench-tool)

## Introduction

The series of notebooks serve as wrapper scripts for running the benchmarks using docker images. In particular, we run the benchmarks of building and searching indexes separately, as we need to sweep through multiple search parameters.

- `collect_build_times.ipynb`: by default it only builds the indexes and collect the build times.
- `collect_search_times.ipynb`: assuming that the indexes have been built, this notebook would sweep through multiple paramter combinations and collect the search times.

## Quickstart

Install dependencies.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks with `papermill`:

```bash
papermill --log-output collect_build_times.ipynb collect_build_times.run1.ipynb

# Dry run
papermill --log-output collect_build_times.ipynb collect_build_times.run1.ipynb -p DRY_RUN 1

# To override parameters with yaml:
papermill --log-output collect_build_times.ipynb collect_build_times.run1.ipynb -y "
DATASETS:
    - "sift-128-euclidean"
"
```

### Recommended workflow and tips

Below is the recommended workflow:

1. Benchmark build times. This process will create the necessary index file for running the searches.
1. Benchmark search times in the `latency` mode. 
    1. The resulting files are copied from `result/search` to `result/search_latency`.
    1. Delete the `result/search` directory before running another search mode. This is recommended because files get overwritten. See [details](https://github.com/rapidsai/raft/issues/2147).
1. Benchmark search times in the `throughput` mode. Note that this could take a long time.
    1. The resulting files are copied from `result/search` to `result/search_throughput`.
1. Upload the following artifacts:
    1. Notebook outputs (build, search in latecy mode, search in throughput mode).
    1. Upload results (`result/build`, `result/search_latency`, `result/search_throughput`).

Tips:

1. Benchmark one dataset at a time.
1. Sometimes benchmarking the search times in throughput mode may hang or exit with error. Monitor GPU usage with `watch nvidia-smi`, and if the GPU usage is stuck at 0%, kill the process. The benchmark should resume.


## Setting up a new host

Prerequisites:

- cuda
- docker
- nvidia container toolkit

Follow the instructions below if any of the prerequisits are missing.

### Update CUDA toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### Install docker

[Instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure

```bash
sudo apt-get -y install containerd
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

### Install mamba (conda)

Install Miniforge, which comes packaged with mamba.

```bash
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
~/conda/bin/conda init
~/conda/bin/mamba init
source ~/.profile
```

**!!Important!!** prevent conda from injecting the `base` virtual environment.

```bash
conda config --set auto_activate_base false
```

Also make sure that the `default` channel for conda is not enabled. Follow instructions [here](https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html#defaults-channels).

```bash
mamba info
```

## Deprecated

### Install the `raft-ann-bench` tool

```bash
mamba create --prefix venv python=3.10
mamba activate ./venv

mamba repoquery search -c rapidsai-nightly raft-ann-bench
mamba install -c rapidsai-nightly -c conda-forge -c nvidia raft-ann-bench=24.02.00a69 cuda-version=12.0*
```

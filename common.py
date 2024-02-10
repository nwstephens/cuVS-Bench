from __future__ import annotations
import string
import subprocess
import pathlib

# Local dataset path
DATASET_PATH = "./benchmarks"
# This should be the path relative in the docker image.
ALGO_CONFIG_DIR = "/data/benchmarks/conf/algos"
CUDA12_DOCKER_IMAGE = "rapidsai/raft-ann-bench:24.02a-cuda12.0-py3.10"
CUDA11_DOCKER_IMAGE = "rapidsai/raft-ann-bench:24.02a-cuda11.8-py3.10"

# Datasets
MNIST_784_EUCLIDEAN = "mnist-784-euclidean"
GLOVE_100_INNER = "glove-100-inner"
SIFT_128_EUCLIDEAN = "sift-128-euclidean"
DEEP_10M_INNER = "deep-image-96-inner"
DEPP_100M = "deep-100M"
GIST_960_EUCLIDEAN = "gist-960-euclidean"
WIKI_ALL_1M = "wiki_all_1M"
WIKI_ALL_10M = "wiki_all_10M"
WIKI_ALL_88M = "wiki_all_88M"


# Algorithms
FAISS_GPU_FLAT = "faiss_gpu_flat"
FAISS_GPU_IVF_FLAT = "faiss_gpu_ivf_flat"
FAISS_GPU_IVF_PQ = "faiss_gpu_ivf_pq"
FAISS_CPU_FLAT = "faiss_cpu_flat"
FAISS_CPU_IVF_FLAT = "faiss_cpu_ivf_flat"
FAISS_CPU_IVF_PQ = "faiss_cpu_ivf_pq"
GGNN = "ggnn"
HNSWLIB = "hnswlib"
RAFT_BRUTE_FORCE = "raft_brute_force"
RAFT_CAGRA = "raft_cagra"
RAFT_IVF_FLAT = "raft_ivf_flat"
RAFT_IVF_PQ = "raft_ivf_pq"
RAFT_CAGRA_HNSWLIB = "raft_cagra_hnswlib"


# Docker command templates
GPU_DOCKER_CMD_TEMPLATE = string.Template(
    """set -x
docker run --rm -u $(id -u) \
    --gpus all \
    --entrypoint /bin/bash \
    --workdir /data/benchmarks \
    -v ${DATASET_PATH}:/data/benchmarks \
    ${DOCKER_IMAGE} -c '${CONTAINER_CMD}'
    """
)

CPU_DOCKER_CMD_TEMPLATE = string.Template(
    """set -x
docker run --rm -u $(id -u) \
    --entrypoint /bin/bash \
    --workdir /data/benchmarks \
    -v ${DATASET_PATH}:/data/benchmarks \
    ${DOCKER_IMAGE} -c '${CONTAINER_CMD}'
    """
)

DOWNLOAD_CMD_TEMPLATE = string.Template(
    """eval "$(conda shell.bash hook)"
    python -m raft-ann-bench.get_dataset --dataset ${DATASET} ${NORMALIZE}"""
)

BUILD_CMD_TEMPLATE = string.Template(
    """eval "$(conda shell.bash hook)"
    python -m raft-ann-bench.run --build \
        --dataset ${DATASET} --algorithms ${ALGORITHMS} ${EXTRA_ARGS}"""
)

SEARCH_CMD_TEMPLATE = string.Template(
    """eval "$(conda shell.bash hook)"
    python -m raft-ann-bench.run --search --search-mode ${SEARCH_MODE} \
        --dataset ${DATASET} --algorithms ${ALGORITHMS} \
        --batch-size ${BATCH_SIZE} --count ${COUNT} ${EXTRA_ARGS}"""
)

EXPORT_DATA_CMD_TEMPLATE = string.Template(
    """eval "$(conda shell.bash hook)"
    python -m raft-ann-bench.data_export --dataset ${DATASET}
    """
)

PLOT_CMD_TEMPLATE = string.Template(
    """eval "$(conda shell.bash hook)"
    python -m raft-ann-bench.plot --mode ${SEARCH_MODE} --dataset ${DATASET} \
        --batch-size ${BATCH_SIZE} --count ${COUNT} ${EXTRA_ARGS}"""
)


def download_wiki(dataset_path: str | pathlib.Path, size: str):
    """
    Download and extract the wiki dataset.

    Args:
        dataset_path (str or pathlib.Path): The path to the dataset directory.
        size (str): The size of the dataset. One of "1M", "10M", or "88M".
    """
    if size not in ("1M", "10M", "88M"):
        raise ValueError("size must be one of '1M', '10M', or '88M'.")

    dataset_name = f"wiki_all_{size}"
    tar_path = dataset_path / f"{dataset_name}.tar"
    if not tar_path.exists():
        cmd = string.Template(
            """
            pushd ${DATASET_PATH}
            wget https://data.rapids.ai/raft/datasets/${DATASET_NAME}/${DATASET_NAME}.tar
            mkdir -p ${DATASET_NAME}
            tar -xvf ${DATASET_NAME}.tar -C ${DATASET_NAME}
            popd
        """
        ).safe_substitute(DATASET_PATH=dataset_path, DATASET_NAME=dataset_name)
        subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, text=True)


def download_wiki_1M(dataset_path: str | pathlib.Path):
    download_wiki(dataset_path, "1M")


def download_wiki_10M(dataset_path: str | pathlib.Path):
    download_wiki(dataset_path, "10M")


def download_wiki_88M(dataset_path: str | pathlib.Path):
    download_wiki(dataset_path, "88M")

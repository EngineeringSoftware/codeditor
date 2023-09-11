#!/bin/bash

# This script prepares a conda environment for running/developing
# Codeditor, with GPU support if an Nvidia GPU is detected.
#
# Requires conda to be installed and available in PATH.
#
# Usage:
#   ./prepare-conda-env.sh
#   # after the script finishes, activate the environment:
#   conda activate deltr
#
# Usage with options:
#   ./prepare-conda-env.sh [cuda_version] [env_name] [conda_path]
#   # cuda_version: {cpu,10.2,11.3,11.6,system} the CUDA toolkit version for PyTorch (default: "11.6" if Nvidia GPU is available detected by nvidia-smi, "cpu" otherwise)
#   # env_name: name of the conda environment to create (default: deltr)
#   # conda_path: path to conda.sh (default: automatically detected)


_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

function get_conda_path() {
        local conda_exe=$(which conda)
        if [[ -z ${conda_exe} ]]; then
                echo "Fail to detect conda! Have you installed Anaconda/Miniconda?" 1>&2
                exit 1
        fi

        echo "$(dirname ${conda_exe})/../etc/profile.d/conda.sh"
}

function get_gpu_avail() {
        if [[ -z $(which nvidia-smi) ]]; then
                echo "cpu"
        else
                echo "gpu"
        fi
}

function get_cuda_version() {
        if [[ -z $(which nvcc) ]]; then
                echo "cpu"
        else
                echo "$(nvcc -V | grep "release" | sed -E "s/.*release ([^,]+),.*/\1/")"
        fi
}

readonly PYTORCH_V=1.12.1
readonly TORCHTEXT_V=0.13.1

function prepare_conda_env() {
        local cuda_version=$1; shift  # cpu|system|10.2|11.3|11.6
        local env_name=${1:-deltr}; shift
        local conda_path=$1; shift

        set -e
        if [[ -z ${cuda_version} ]]; then
                if [[ $(get_gpu_avail) == "gpu" ]]; then
                        # by default, use newer cuda version for better compatibility with newer GPUs
                        cuda_version="11.6"
                else
                        cuda_version="cpu"
                fi
        fi
        if [[ -z ${conda_path} ]]; then
                conda_path=$(get_conda_path)
        fi
        echo ">>> Preparing conda environment \"${env_name}\"; cuda_version: ${cuda_version}; conda path: ${conda_path}"
        
        # Preparation
        source ${conda_path}
        conda env remove --name $env_name
        conda create --name $env_name python=3.9 pip -y
        conda activate $env_name

        # Install Pytorch
        case ${cuda_version} in
        cpu)
                conda install -y pytorch=${PYTORCH_V} torchtext=${TORCHTEXT_V} cpuonly -c pytorch
                ;;
        10.2)
                conda install -y pytorch=${PYTORCH_V} torchtext=${TORCHTEXT_V} cudatoolkit=10.2 -c pytorch
                ;;
        11.3)
                conda install -y pytorch=${PYTORCH_V} torchtext=${TORCHTEXT_V} cudatoolkit=11.3 -c pytorch
                ;;
        11.6)
                conda install -y pytorch=${PYTORCH_V} torchtext=${TORCHTEXT_V} cudatoolkit=11.6 -c pytorch -c conda-forge
                ;;
        system)
                local sys_cuda_version=$(get_cuda_version)
                case ${sys_cuda_version} in
                10.2)
                        echo ">>> Found system cuda ${sys_cuda_version}, attemping to install pytorch with pip..."
                        pip install torch==${PYTORCH_V}+cu102 torchtext==${TORCHTEXT_V} --extra-index-url https://download.pytorch.org/whl/cu102
                        ;;
                11.3)
                        echo ">>> Found system cuda ${sys_cuda_version}, attemping to install pytorch with pip..."
                        pip install torch==${PYTORCH_V}+cu113 torchtext==${TORCHTEXT_V} --extra-index-url https://download.pytorch.org/whl/cu113
                        ;;
                11.6)
                        echo ">>> Found system cuda ${sys_cuda_version}, attemping to install pytorch with pip..."
                        pip install torch==${PYTORCH_V}+cu116 torchtext==${TORCHTEXT_V} --extra-index-url https://download.pytorch.org/whl/cu116
                        ;;
                *)
                        echo ">>> [ERROR] Not found compatible system cuda (detected ${sys_cuda_version})!"
                        return
                        ;;
                esac
                ;;
        *)
                echo ">>> [ERROR] cuda_version=${cuda_version} is not supported!"
                return
                ;;
        esac

}


prepare_conda_env "$@"
#!/usr/bin/env bash
#SBATCH --time=01-00:00:00
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --output=out_local_train.txt

export PYTHONPATH="${PYTHONPATH}:${PWD}/src:${PWD}/src/pt"

python3 ./main.py


#!/usr/bin/env bash
#SBATCH --time=01-00:00:00
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --output=out_local_train.txt

export PYTHONPATH="${PYTHONPATH}:${PWD}/code:${PWD}/code/pt"

python3 ./code/pt/learners/local_mammo_learner.py 'pipelines' 'resnet'


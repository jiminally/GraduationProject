#!/usr/bin/bash

#SBATCH -J VTG_Full_Refine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -w moana-y5
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

mkdir -p logs

pwd
which python
hostname

source ~/.bashrc
conda activate vtg-gpt

cd Baichuan2
python rephrase_query.py

exit 0
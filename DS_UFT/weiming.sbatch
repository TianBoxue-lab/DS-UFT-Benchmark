#!/bin/bash
# Specify job name for identification in queue preview
#SBATCH --job-name=human
# Specify job output and error files
#SBATCH -o ./log_0.5_combine/log_%j.out
#SBATCH -e ./log_0.5_combine/log_%j.err
# Specify partition for GPU jobs
#SBATCH --partition=batch
# Total number of GPUs for the task
#SBATCH --gres=gpu:8
# Memory per GPU (do not modify unless necessary)
#SBATCH --mem-per-gpu=60000
# Number of tasks
#SBATCH --ntasks=1
# CPUs per GPU (do not modify unless necessary)
#SBATCH --cpus-per-gpu=12

# Output basic job information, modify as needed
echo SLURM_JOB_ID=$SLURM_JOB_ID
echo SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES
echo SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
echo SLURM_JOB_GPUS=$SLURM_JOB_GPUS
echo SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE

# Record subtask start time
echo start time:$(date)
# Execute task
conda activate /software/env/weiming/esm
python /homes/Tianlab/weiming/esm/tune_training.py
# Record subtask end time
echo finish time:$(date)

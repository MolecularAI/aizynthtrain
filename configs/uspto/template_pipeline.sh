#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g

source ~/.bashrc 
conda activate aizynthtrain
unset JUPYTER_PATH

python -m aizynthtrain.pipelines.template_pipeline run --config template_pipeline_config.yml --max-workers 32 --max-num-splits 200
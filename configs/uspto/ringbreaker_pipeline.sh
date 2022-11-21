#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g

source ~/.bashrc 
conda activate aizynthtrain
unset JUPYTER_PATH

python -m aizynthtrain.pipelines.expansion_model_pipeline run --config ringbreaker_model_pipeline_config.yml  --max-workers 8 --max-num-splits 200
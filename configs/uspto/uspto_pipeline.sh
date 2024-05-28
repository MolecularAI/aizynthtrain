conda activate rxn-env

python -m rxnutils.data.uspto.preparation_pipeline run --nbatches 200  --max-workers 32 --max-num-splits 200

conda activate rxnmapper

python -m rxnutils.data.uspto.mapping_pipeline run --nbatches 200  --max-workers 64 --max-num-splits 200

#!/usr/bin/env bash

for DATASET in ecthr; do
#    MODEL_PATH_BASE="../contrastive/s3-link/clean"
    MODEL_PATH="experiments/models/${DATASET}/allenai/longformer-base-4096"

    CUDA_VISIBLE_DEVICES=0,1,2,3 allennlp train configs/${DATASET}_hpc.jsonnet \
     -s ${MODEL_PATH} -f \
     --include-package=allennlp_lib

#    mkdir ${MODEL_PATH}/encodings
#    find ${MODEL_PATH} -name "*.th" -delete

    python scripts/cache_linear_classifier.py --model-path=${MODEL_PATH}
done

#!/usr/bin/env bash

GPU_IDX=0

for DATASET in ecthr; do
#    MODEL_PATH_BASE="../contrastive/s3-link/clean"
    MODEL_PATH="experiments/models/${DATASET}/allenai/longformer-base-4096"
    INPUT_PATH="data/${DATASET}/simple_val.jsonl"

    CUDA_VISIBLE_DEVICES=${GPU_IDX} allennlp evaluate configs/${DATASET}.jsonnet \
     --include-package=allennlp_lib \ 
     ${MODEL_PATH} ${INPUT_PATH}

#    mkdir ${MODEL_PATH}/encodings
#    find ${MODEL_PATH} -name "*.th" -delete

    python3 scripts/cache_linear_classifier.py --model-path=${MODEL_PATH}
done

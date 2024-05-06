#!/usr/bin/env bash

GPU_IDX=0

#3e-4_0.2_50 3e-4_0.2_100 3e-4_0.2_200 3e-4_0.2_300 3e-4_0.3_50 3e-4_0.3_100 3e-4_0.3_200 3e-4_0.3_300 3e-4_0.4_50 3e-4_0.4_100 3e-4_0.4_200 3e-4_0.4_300 3e-5_0.2_50 3e-5_0.2_100 3e-5_0.2_200 3e-5_0.2_300 3e-5_0.3_50 3e-5_0.3_100 3e-5_0.3_200 3e-5_0.3_300 3e-5_0.4_50 3e-5_0.4_100 3e-5_0.4_200

for config_file in 3e-5_0.4_300 3e-6_0.2_50 3e-6_0.2_100 3e-6_0.2_200 3e-6_0.2_300 3e-6_0.3_50 3e-6_0.3_100 3e-6_0.3_200 3e-6_0.3_300 3e-6_0.4_50 3e-6_0.4_100 3e-6_0.4_200 3e-6_0.4_300; do
#    MODEL_PATH_BASE="../contrastive/s3-link/clean"
    MODEL_PATH="experiments/models/ecthr/allenai/longformer-base-4096"

    CUDA_VISIBLE_DEVICES=${GPU_IDX} allennlp train configs/ecthr_${config_file}.jsonnet \
     -s ${MODEL_PATH} -f \
     --include-package=allennlp_lib

#    mkdir ${MODEL_PATH}/encodings
#    find ${MODEL_PATH} -name "*.th" -delete

    python3 scripts/cache_linear_classifier.py --model-path=${MODEL_PATH} --include-package=allennlp_lib

    cp experiments/models/ecthr/allenai/longformer-base-4096/out.log ${config_file}.txt
done

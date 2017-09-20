#!/bin/bash

# MODIFY PATH for YOUR SETTING
 
PATH_TO_MATLAB=/PATH_TO_YOUR_MATLAB
PATH_TO_SCRIPT=./scripts/camvid/test_segmentation_camvid.py
PATH_TO_BN_SCRIPT=./scripts/camvid/compute_bn_statistics_camvid.py
PATH_TO_MAT_SCRIPT=./scripts/camvid/compute_test_results_camvid.m
PATH_TO_TRAIN_MODEL=./models/camvid/train_camvid_gate.prototxt
PATH_TO_TEST_MODEL=./models/camvid/test_camvid_gate.prototxt
PATH_TO_TRAIN_WEIGHT=./models/camvid/gated_frnet_camvid_test_wo_bn.caffemodel
PATH_TO_TEST_WEIGHT=./models/camvid/Inference_weight_camvid/test_weights.caffemodel
PATH_TO_SAVE_TEST_WEIGHT=./models/camvid/Inference_weight_camvid/

python ${PATH_TO_BN_SCRIPT} ${PATH_TO_TRAIN_MODEL} ${PATH_TO_TRAIN_WEIGHT} ${PATH_TO_SAVE_TEST_WEIGHT}
python ${PATH_TO_SCRIPT} --model ${PATH_TO_TEST_MODEL} --weights ${PATH_TO_TEST_WEIGHT} --iter 233
${PATH_TO_MATLAB} -nodisplay -r "run ${PATH_TO_MAT_SCRIPT}; exit"

#!/bin/bash

CAFFE_ROOT_DIR=caffe-gfrnet/build/tools/caffe
PATH_TO_SOLVER=./models/camvid/solver_camvid_gate.prototxt
PATH_TO_MODEL=./models/VGG_ILSVRC_16_layers.caffemodel

DEV_ID=1

./${CAFFE_ROOT_DIR} train -solver ${PATH_TO_SOLVER} -weights ${PATH_TO_MODEL} -gpu ${DEV_ID}


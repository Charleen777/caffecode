#!/usr/bin/env sh

LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`-deeplab_lstm1.log
CAFFE=./build/tools/caffe

$CAFFE train --solver=prototxt/deeplab_lstm1/solver.prototxt --weights=prototxt/deeplab_LargeFOV/train2_iter_8000.caffemodel 2>&1 | tee $LOG


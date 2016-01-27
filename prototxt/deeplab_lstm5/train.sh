#!/usr/bin/env sh

LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`-deeplab_lstm5.log
CAFFE=./build/tools/caffe

$CAFFE train --solver=prototxt/deeplab_lstm5/solver.prototxt --weights=models/vgg16_20M.caffemodel 2>&1 | tee $LOG


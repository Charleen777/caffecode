#!/usr/bin/env sh

LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`-vgg5_lstm1.log
CAFFE=./build/tools/caffe

$CAFFE train --solver=prototxt/vgg5_lstm1/solver.prototxt --weights=models/VGG_ILSVRC_16_layers.caffemodel 2>&1 | tee $LOG


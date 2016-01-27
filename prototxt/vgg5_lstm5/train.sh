#!/usr/bin/env sh

LOG=logfile/train-`date +%Y-%m-%d-%H-%M-%S`-vgg5_lstm5.log
CAFFE=./build/tools/caffe

$CAFFE train --solver=prototxt/vgg5_lstm5/solver.prototxt --weights=models/VGG_ILSVRC_16_layers.caffemodel --gpu=3 2>&1 | tee $LOG


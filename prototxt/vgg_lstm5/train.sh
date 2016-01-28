#!/usr/bin/env sh

LOG=logfile/train-`date +%Y-%m-%d-%H-%M-%S`-vgg_lstm5.log
CAFFE=./build/tools/caffe

$CAFFE train --solver=prototxt/vgg_lstm5/solver.prototxt --weights=models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel --gpu=0 2>&1 | tee $LOG


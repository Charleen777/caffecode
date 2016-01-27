#!/usr/bin/env sh

LOG=log/test-`date +%Y-%m-%d-%H-%M-%S`-fcn_lstm1_interp.log
CAFFE=./build/tools/caffe

$CAFFE test --model=prototxt/fcn_lstm1/train.prototxt --weights=snapshots/fcn_lstm1_iter_$1.caffemodel --iterations=200   2>&1 | tee $LOG
#--gpu=0

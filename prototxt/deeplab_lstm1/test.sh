#!/usr/bin/env sh

LOG=log/test-`date +%Y-%m-%d-%H-%M-%S`-deeplab_lstm1.log
CAFFE=./build/tools/caffe

$CAFFE test --model=prototxt/train.prototxt --weights=snapshots/deeplab_lstm1_iter_$1.caffemodel --iterations=200  2>&1 | tee $LOG

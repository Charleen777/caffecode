#!/usr/bin/env sh

LOG=log/test-`date +%Y-%m-%d-%H-%M-%S`-vgg5_lstm1.log
CAFFE=./build/tools/caffe

$CAFFE test --model=prototxt/vgg5_lstm1/test.prototxt --weights=snapshots/vgg5_lstm1_iter_$1.caffemodel --iterations=200 --gpu=0  2>&1 | tee $LOG

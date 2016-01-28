#!/usr/bin/env sh

LOG=logfile/test-`date +%Y-%m-%d-%H-%M-%S`_lstm5.log
CAFFE=./build/tools/caffe

$CAFFE test --model=prototxt/vgg_lstm5/train_val.prototxt --weights=snapshots/vgg_lstm5_iter_$1.caffemodel --iterations=200 --gpu=2  2>&1 | tee $LOG

#!/usr/bin/env sh

LOG=logfile/test-`date +%Y-%m-%d-%H-%M-%S`_vgg5_lstm5_noshare.log
CAFFE=./build/tools/caffe

$CAFFE test --model=prototxt/vgg5_lstm5_noshare/train_val.prototxt --weights=snapshots/vgg5_lstm5_noshare_iter_$1.caffemodel --iterations=200 --gpu=2  2>&1 | tee $LOG

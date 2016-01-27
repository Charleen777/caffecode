#!/usr/bin/env sh

LOG=log/train-`date +%Y-%m-%d-%H-%M-%S`-fcn_lstm1.log
CAFFE=./build/tools/caffe

$CAFFE train --solver=prototxt/fcn_lstm1/solver.prototxt --weights=models/fcn-16s-sift-flow.caffemodel 2>&1 | tee $LOG


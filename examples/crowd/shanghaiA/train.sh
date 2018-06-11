#!/usr/bin/env sh

TOOLS=./build/tools
protopath=./examples/crowd/code/shanghaiA

$TOOLS/caffe train \
    --solver=$protopath/solver_n1_1.prototxt\
    --weights=$protopath/VGG_ILSVRC_16_layers.caffemodel
    #--weights=./examples/crowd/code/shanghaiB/result/network_vgg_v1_iter_247738.caffemodel
    #--weights=$protopath/model/network_vgg_avg_iter_361800.caffemodel
    #--weights=$protopath/model/network_vgg_0.001_iter_372600.caffemodel
    #--snapshot=$protopath/model/network/network_vgg_v1_iter_1026000.solverstate
    #--weights=$protopath/model/model2/fold1/network_iter_252000.caffemodel

#$TOOLS/caffe train \
    #--solver=$protopath/solver_n1_2.prototxt \
    #--snapshot=$protopath/model/network_iter_2160000.solverstate
    #--weights=$protopath/model/network_iter_72279.caffemodel



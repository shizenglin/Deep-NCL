#!/usr/bin/env sh

TOOLS=./build/tools
protopath=./examples/crowd/code/shanghaiB

$TOOLS/caffe train \
    --solver=$protopath/solver_n1_1.prototxt\
    --weights=./examples/crowd/code/shanghaiA/vgg64-result/64-1.01/network_vgg_1.01_iter_140400.caffemodel
    #--weights=$protopath/result/network_vgg_v1_iter_247738.caffemodel
    #--weights=$protopath/VGG_ILSVRC_16_layers.caffemodel
    #--weights=$protopath/model/network3/network_vgg_v1_iter_247738.caffemodel
    #--snapshot=$protopath/network_iter_1339200.solverstate
    #--weights=$protopath/model/model2/fold1/network_iter_252000.caffemodel
    #--weights=$protopath/VGG_ILSVRC_16_layers.caffemodel

#$TOOLS/caffe train \
    #--solver=$protopath/solver_n1_2.prototxt \
    #--snapshot=$protopath/model/network_iter_2160000.solverstate
    #--weights=$protopath/model/network_iter_72279.caffemodel



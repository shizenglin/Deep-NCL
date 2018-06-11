#include <vector>
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mae_loss_layer_v2.hpp"

namespace caffe {

template <typename Dtype>
void MAELossV2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_mul(
      count,
      bottom[0]->gpu_data(),
      bottom[2]->gpu_data(),
      multi1_.mutable_gpu_data());
  caffe_gpu_mul(
      count,
      bottom[1]->gpu_data(),
      bottom[2]->gpu_data(),
      multi2_.mutable_gpu_data());
  Dtype asum1,asum2;
  caffe_gpu_asum(count, multi1_.gpu_data(), &asum1);
  caffe_gpu_asum(count, multi2_.gpu_data(), &asum2);
  Dtype loss = abs(asum1-asum2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MAELossV2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(MAELossV2Layer);

}  // namespace caffe

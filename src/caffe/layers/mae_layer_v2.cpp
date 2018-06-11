#include <vector>
#include <math.h>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mae_loss_layer_v2.hpp"

namespace caffe {

template <typename Dtype>
void MAELossV2Layer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  multi1_.ReshapeLike(*bottom[0]);
  multi2_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MAELossV2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_mul(
      count,
      bottom[0]->cpu_data(),
      bottom[2]->cpu_data(),
      multi1_.mutable_cpu_data());
  caffe_mul(
      count,
      bottom[1]->cpu_data(),
      bottom[2]->cpu_data(),
      multi2_.mutable_cpu_data());
  Dtype asum1 =  caffe_cpu_asum(count, multi1_.cpu_data());
  Dtype asum2 =  caffe_cpu_asum(count, multi2_.cpu_data());
  Dtype loss = abs(asum1-asum2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MAELossV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MAEV2Layer);
#endif

INSTANTIATE_CLASS(MAELossV2Layer);
REGISTER_LAYER_CLASS(MAELossV2);

}  // namespace caffe

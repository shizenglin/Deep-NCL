#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ncl_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void NCLLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype lambda = this->layer_param_.ncl_loss_param().lambda();
  int net_num = this->layer_param_.ncl_loss_param().net_num();
  Dtype norm = Dtype(1)/(Dtype(net_num)*Dtype(net_num)-lambda*(Dtype(net_num)-Dtype(1))*(Dtype(net_num)-Dtype(1)));
  caffe_gpu_sub(
      count,
      bottom[2]->gpu_data(),
      bottom[0]->gpu_data(),
      avgdiff_.mutable_gpu_data());

  caffe_gpu_scal(count,lambda*(Dtype(net_num)-Dtype(1)),avgdiff_.mutable_gpu_data());
  caffe_gpu_axpby(
          count,              // count
          Dtype(net_num)*Dtype(net_num),                              // alpha
          bottom[1]->gpu_data(),                   // a
          Dtype(0),                           // beta
          label_.mutable_gpu_data());  // b

  caffe_gpu_sub(
      count,
      label_.gpu_data(),
      avgdiff_.gpu_data(),
      label_.mutable_gpu_data());
  caffe_gpu_scal(count,norm,label_.mutable_gpu_data());
  
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      label_.gpu_data(),
      diff_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NCLLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NCLLossLayer);

}  // namespace caffe

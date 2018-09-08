#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ncl_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void NCLLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  avgdiff_.ReshapeLike(*bottom[0]);
  label_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NCLLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype lambda = this->layer_param_.ncl_loss_param().lambda();
  int net_num = this->layer_param_.ncl_loss_param().net_num();
  Dtype norm = Dtype(1)/(Dtype(net_num)*Dtype(net_num)-lambda*(Dtype(net_num)-Dtype(1))*(Dtype(net_num)-Dtype(1)));
  caffe_sub(
      count,
      bottom[2]->cpu_data(),
      bottom[0]->cpu_data(),
      avgdiff_.mutable_cpu_data());

  caffe_scal(count,lambda*(Dtype(net_num)-Dtype(1)),avgdiff_.mutable_cpu_data());
  caffe_cpu_axpby(
          count,              // count
          Dtype(net_num)*Dtype(net_num),                              // alpha
          bottom[1]->cpu_data(),                   // a
          Dtype(0),                           // beta
          label_.mutable_cpu_data());  // b

  caffe_sub(
      count,
      label_.cpu_data(),
      avgdiff_.cpu_data(),
      label_.mutable_cpu_data());
  caffe_scal(count,norm,label_.mutable_cpu_data());
  
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      label_.cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NCLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NCLLossLayer);
#endif

INSTANTIATE_CLASS(NCLLossLayer);
REGISTER_LAYER_CLASS(NCLLoss);

}  // namespace caffe

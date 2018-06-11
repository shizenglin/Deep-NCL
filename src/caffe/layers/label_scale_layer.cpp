#include <vector>
#include <math.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/label_scale_layer.hpp"

namespace caffe {

template <typename Dtype>
void LabelScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  //Dtype asum1 =  caffe_cpu_asum(count, bottom[0]->cpu_data());
  Dtype asum1 =  caffe_cpu_asum(count, bottom[1]->cpu_data());
  //int const num= bottom[1]->num();
  //int const channels=bottom[1]->channels();
  int const height=bottom[1]->height();
  int const width=bottom[1]->width();
  int const height1=bottom[0]->height();
  int const width1=bottom[0]->width();
  //Blob<Dtype> newlabel_(vector<int>(height, width));
  const Dtype * templabel_=bottom[1]->cpu_data();
  //templabel_.reshape(vector<int>(bottom[1]->height(), bottom[1]->width());
  
  cv::Mat label_image = cv::Mat::zeros(height, width, CV_32FC1);
  cv::Mat cv_img;
  int image_offset = 0;
  int h,w;
  for (h = 0; h < height; ++h) {
    for (w = 0; w < width; ++w) {
      label_image.at<cv::Vec3b>(h, w)[0] = templabel_[image_offset];
      ++image_offset;
    }
  }
  cv::resize(label_image, cv_img, cv::Size(height1, width1));
  Dtype asum2=cv::sum(cv_img)[0];
  Dtype norm = asum1/asum2;
  
  top[0]->ReshapeLike(*bottom[0]);
  Dtype * templabel1_=top[0]->mutable_cpu_data();
  image_offset = 0;
  for (h = 0; h < height1; ++h) {
    for (w = 0; w < width1; ++w) {
      templabel1_[image_offset] = static_cast<Dtype>(label_image.at<cv::Vec3b>(h, w)[0]*norm);
      ++image_offset;
    }
  } 
  
}

#ifdef CPU_ONLY
STUB_GPU(LabelScaleLayer);
#endif

INSTANTIATE_CLASS(LabelScaleLayer);
REGISTER_LAYER_CLASS(LabelScale);

}  // namespace caffe

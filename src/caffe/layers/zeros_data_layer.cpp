#include <algorithm>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void ZerosDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ZerosDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape;
  const int shape_size = this->layer_param_.zeros_data_param().shape_size();
  for (int i = 0; i < shape_size; ++i) {
    shape.push_back(this->layer_param_.zeros_data_param().shape(i));
    CHECK_LE(0, shape[i]) << "All numpy data dimensions must be non-zero";
  }
  if(shape_size > 2){
  	top[0]->Reshape(shape[0],shape[1],shape[2],shape[3]);
  } else {
     vector<int> top_shape(2);
     top_shape[0] = shape[0];
     top_shape[1] = shape[1];
  	 top[0]->Reshape(top_shape);
  }

}

template <typename Dtype>
void ZerosDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  memset(top_data,0,sizeof(Dtype)*top[0]->count());
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ZerosDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ZerosDataLayer);
REGISTER_LAYER_CLASS(ZerosData);
}  // namespace caffe

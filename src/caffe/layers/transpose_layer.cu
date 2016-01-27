#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (this->layer_param_.transpose_param().transposetype()) {
  case TransposeParameter_TransposeType_LSTMINPUT:
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		const int spatial_dim = bottom[0]->width() * bottom[0]->height();
		caffe_gpu_transpose(bottom[0]->channels(), spatial_dim,
		    bottom_data + bottom[0]->offset(n),
		    top_data + top[0]->offset(n * spatial_dim));
	  }
	break;
  case TransposeParameter_TransposeType_LSTMOUTPUT:
	 Forward_cpu(bottom, top);
	  /*for (int n = 0; n < bottom[0]->num(); ++n) {
		caffe_gpu_transpose(bottom[0]->channels() * bottom[0]->height(), bottom[0]->width(),
		    bottom_data + bottom[0]->offset(n),
		    top_data + top[0]->offset(n * bottom[0]->width()));
	  }*/
    break;
  case TransposeParameter_TransposeType_GRIDLSTM:
	 Forward_cpu(bottom, top);
    break;
  case TransposeParameter_TransposeType_GRIDLSTMSUM:
         Forward_cpu(bottom, top);
    break; 
 }
///////
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  /*Backward_cpu(top, propagate_down, bottom);*/
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  switch (this->layer_param_.transpose_param().transposetype()) {
  case TransposeParameter_TransposeType_LSTMINPUT:
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		const int spatial_dim = bottom[0]->width() * bottom[0]->height();
		caffe_gpu_transpose(bottom[0]->width() * bottom[0]->height(),
		    bottom[0]->channels(), top_diff + top[0]->offset(n * spatial_dim),
		    bottom_diff + bottom[0]->offset(n));
	  }
	break;
  case TransposeParameter_TransposeType_LSTMOUTPUT:
		Backward_cpu(top, propagate_down, bottom);
//////
	  /*for (int n = 0; n < bottom[0]->num(); ++n) {
		caffe_gpu_transpose(bottom[0]->width(),
		    bottom[0]->channels() * bottom[0]->height(), top_diff + top[0]->offset(n * bottom[0]->width()),
		    bottom_diff + bottom[0]->offset(n));
	  }*/
    break;
   case TransposeParameter_TransposeType_GRIDLSTM:
		Backward_cpu(top, propagate_down, bottom);
	break;
    case TransposeParameter_TransposeType_GRIDLSTMSUM:
                Backward_cpu(top, propagate_down, bottom);
        break;
  }
//////
}


INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);

}  // namespace caffe

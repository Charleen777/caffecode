#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2), top_shape2(4);
  switch (this->layer_param_.transpose_param().transposetype()) {
  case TransposeParameter_TransposeType_LSTMINPUT:
  	top_shape[0] = bottom[0]->num() * bottom[0]->height() * bottom[0]->width();
  	top_shape[1] = bottom[0]->channels();
  	top[0]->Reshape(top_shape);
  	CHECK_EQ(top[0]->count(), bottom[0]->count());
    break;
  case TransposeParameter_TransposeType_LSTMOUTPUT:
    top_shape2[0] = this->layer_param_.transpose_param().batch_size();
    top_shape2[1] = bottom[0]->shape(1); //channels
    top_shape2[2] = this->layer_param_.transpose_param().height(); //height
    top_shape2[3] = bottom[0]->shape(0) / top_shape2[0] / top_shape2[2];  //width
    top[0]->Reshape(top_shape2);
  	CHECK_EQ(top[0]->count(), bottom[0]->count());
    break;
  case TransposeParameter_TransposeType_GRIDLSTM:
    top_shape[0] = bottom[0]->num();
    top_shape[1] = bottom[0]->channels();
    top[0]->Reshape(top_shape);
  	CHECK_EQ(top[0]->count(), bottom[0]->count());
  
   break;
  case TransposeParameter_TransposeType_GRIDLSTMSUM:
    int local_connected_num = this->layer_param_.transpose_param().local_connected_num();
    top_shape[0] = bottom[0]->num();
    top_shape[1] = bottom[0]->channels()/local_connected_num;
    top[0]->Reshape(top_shape);  
   break;
  }
  
  
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  
  int spatial_dim, channels, batch_size, local_connected_num, height, each_channel;
  int trans_offset[8];
 switch (this->layer_param_.transpose_param().transposetype()) {
  case TransposeParameter_TransposeType_LSTMINPUT:
	  spatial_dim = bottom[0]->height() * bottom[0]->width();
	  channels = bottom[0]->channels();
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		const int offset = bottom[0]->count() / bottom[0]->num() * n;
		for (int c = 0; c < channels; ++c) {
		  for (int i = 0; i < spatial_dim; ++i) {
		    top_data[offset + i * channels + c]
		        = bottom_data[offset + c * spatial_dim + i];
		  }
		}
	  }
      break;
   case TransposeParameter_TransposeType_LSTMOUTPUT:
	  spatial_dim = top[0]->height() * top[0]->width();
	  channels = top[0]->channels();
	  for (int n = 0; n < top[0]->num(); ++n) {
		const int offset = top[0]->count() / top[0]->num() * n;
		for (int c = 0; c < channels; ++c) {
		  for (int i = 0; i < spatial_dim; ++i) {
		    top_data[offset + c * spatial_dim + i]
		        = bottom_data[offset + i * channels + c];
		  }
		}
	  }
      break;
	case TransposeParameter_TransposeType_GRIDLSTM:
 		//GridLSTM
  	  batch_size = this->layer_param_.transpose_param().batch_size();
  	  local_connected_num = this->layer_param_.transpose_param().local_connected_num();
          height = this->layer_param_.transpose_param().height();
          spatial_dim = bottom[0]->num() / batch_size;
          channels = bottom[0]->channels();
	  each_channel = channels/local_connected_num;
          trans_offset[0] = -height-1;  trans_offset[1] = -height; trans_offset[2] = -height+1; trans_offset[3] = -1; trans_offset[4] = 1; 
          trans_offset[5] = height-1; trans_offset[6] = height; trans_offset[7] = height+1;

	  //LOG(INFO) << spatial_dim << " "<< each_channel<< " "<< local_connected_num;
      CHECK_EQ(local_connected_num, Dtype(8));// only support 8-local neighborhood
      caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
      //-h-1, -h, -h+1, -1, +1, +h-1, +h, +h+1
      for (int n = 0; n < batch_size; ++n) {
      	const int offset = bottom[0]->count() / batch_size * n;
		for (int i = 0; i < spatial_dim; ++i) {
			for (int ln = 0; ln < local_connected_num; ++ln) {
				int transIndex = i + trans_offset[ln];
                    //LOG(INFO) << transIndex;
				if((transIndex > 0) & (transIndex < spatial_dim)) {				
					for (int c = 0; c < each_channel; ++c){
						top_data[offset + i*channels + ln*each_channel + c] // !!! error: local_connected_num -> each_channel
			 				= bottom_data[offset + transIndex*channels + (local_connected_num - ln - 1)*each_channel + c];
					}
                    //else set original 0
				}
			}					
		}
      }
	  break;
     case TransposeParameter_TransposeType_GRIDLSTMSUM:
 		//GridLSTM
  	  batch_size = this->layer_param_.transpose_param().batch_size();
  	  local_connected_num = this->layer_param_.transpose_param().local_connected_num();
      height = this->layer_param_.transpose_param().height();
      spatial_dim = bottom[0]->num() / batch_size;
      channels = bottom[0]->channels();
	  each_channel = channels/local_connected_num;
	  //LOG(INFO) << spatial_dim << " "<< each_channel<< " "<< local_connected_num;
      trans_offset[0] = -height-1;  trans_offset[1] = -height; trans_offset[2] = -height+1; trans_offset[3] = -1; trans_offset[4] = 1; 
      trans_offset[5] = height-1; trans_offset[6] = height; trans_offset[7] = height+1;

      CHECK_EQ(local_connected_num, Dtype(8));// only support 8-local neighborhood
      caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
      //-h-1, -h, -h+1, -1, +1, +h-1, +h, +h+1
      for (int n = 0; n < batch_size; ++n) {
      	const int offset = bottom[0]->count() / batch_size * n;
		for (int i = 0; i < spatial_dim; ++i) {
			for (int c = 0; c < each_channel; ++c){
				for (int ln = 0; ln < local_connected_num; ++ln) {							
					int transIndex = i + trans_offset[ln];
                    //LOG(INFO) << transIndex;
					if((transIndex > 0) & (transIndex < spatial_dim)) {
						top_data[offset + i*each_channel + c]
			 				+= bottom_data[offset + transIndex*channels + ln*local_connected_num + c];
					}
                    //else set original 0
				}
			}					
		}
      }
	  break;
 }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int spatial_dim, channels, batch_size, local_connected_num, height, each_channel;
   int trans_offset[8];
  switch (this->layer_param_.transpose_param().transposetype()) {
  case TransposeParameter_TransposeType_LSTMINPUT:
	  spatial_dim = bottom[0]->height() * bottom[0]->width();
	  channels = bottom[0]->channels();
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		const int offset = bottom[0]->count() / bottom[0]->num() * n;
		for (int c = 0; c < channels; ++c) {
		  for (int i = 0; i < spatial_dim; ++i) {
		    bottom_diff[offset + c * spatial_dim + i] =
		        top_diff[offset + i * channels + c];
		  }
		}
	  }
      break;
   case TransposeParameter_TransposeType_LSTMOUTPUT:
	  spatial_dim = top[0]->height() * top[0]->width();
	  channels = top[0]->channels();
	  for (int n = 0; n < top[0]->num(); ++n) {
		const int offset = top[0]->count() / top[0]->num() * n;
		for (int c = 0; c < channels; ++c) {
		  for (int i = 0; i < spatial_dim; ++i) {
		    bottom_diff[offset + i * channels + c] =
		        top_diff[offset + c * spatial_dim + i];
		  }
		}
	  }
      break;
	case TransposeParameter_TransposeType_GRIDLSTM:
 		//GridLSTM
  	 batch_size = this->layer_param_.transpose_param().batch_size();
  	 local_connected_num = this->layer_param_.transpose_param().local_connected_num();
      height = this->layer_param_.transpose_param().height();
      spatial_dim = bottom[0]->num() / batch_size;
      channels = bottom[0]->channels();
	  each_channel = channels/local_connected_num;
	  //LOG(INFO) << spatial_dim << " "<< each_channel<< " "<< local_connected_num;

      trans_offset[0] = -height-1;  trans_offset[1] = -height; trans_offset[2] = -height+1; trans_offset[3] = -1; trans_offset[4] = 1; 
      trans_offset[5] = height-1; trans_offset[6] = height; trans_offset[7] = height+1;

     CHECK_EQ(local_connected_num, Dtype(8));// only support 8-local neighborhood
      //-h-1, -h, -h+1, -1, +1, +h-1, +h, +h+1
      for (int n = 0; n < batch_size; ++n) {
      	const int offset = bottom[0]->count() / batch_size * n;
		for (int i = 0; i < spatial_dim; ++i) {
			for (int ln = 0; ln < local_connected_num; ++ln) {
				int transIndex = i + trans_offset[ln];
				if((transIndex > 0) & (transIndex < spatial_dim)) {			
					for (int c = 0; c < each_channel; ++c){
						bottom_diff[offset + transIndex*channels + (local_connected_num - ln - 1)*each_channel + c]						
							= top_diff[offset + i*channels + ln*each_channel + c];
					}
                    //else set original 0
				}

			}					
		}
      }
		break;
     case TransposeParameter_TransposeType_GRIDLSTMSUM:
 		//GridLSTM
  	  batch_size = this->layer_param_.transpose_param().batch_size();
  	  local_connected_num = this->layer_param_.transpose_param().local_connected_num();
      height = this->layer_param_.transpose_param().height();
      spatial_dim = bottom[0]->num() / batch_size;
      channels = bottom[0]->channels();
	  each_channel = channels/local_connected_num;
      
      trans_offset[0] = -height-1;  trans_offset[1] = -height; trans_offset[2] = -height+1; trans_offset[3] = -1; trans_offset[4] = 1; 
      trans_offset[5] = height-1; trans_offset[6] = height; trans_offset[7] = height+1;


      //LOG(INFO) << spatial_dim << " "<< each_channel<< " "<< local_connected_num;
      CHECK_EQ(local_connected_num, Dtype(8));// only support 8-local neighborhood
      //-h-1, -h, -h+1, -1, +1, +h-1, +h, +h+1
      for (int n = 0; n < batch_size; ++n) {
      	const int offset = bottom[0]->count() / batch_size * n;
		for (int i = 0; i < spatial_dim; ++i) {
			for (int c = 0; c < each_channel; ++c){
				for (int ln = 0; ln < local_connected_num; ++ln) {							
 					int transIndex = i + trans_offset[ln];
					if((transIndex > 0) & (transIndex < spatial_dim)) {
						bottom_diff[offset + transIndex*channels + ln*local_connected_num + c]						
							= top_diff[offset + i*each_channel + c];
					}
                    //else set original 0
				}

			}					
		}
      }
		break;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe

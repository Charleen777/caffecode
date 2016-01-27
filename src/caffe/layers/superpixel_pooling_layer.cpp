#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template<typename Dtype>
void SuperpixelPoolingLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
}

template<typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	superpixel_num_ =
			this->layer_param_.superpixel_pooling_param().superpixel_num();
	pooled_height_ = superpixel_num_;
	pooled_width_ = 1;

	top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
	if (top.size() > 1) {
		top[1]->ReshapeLike(*top[0]);
	}
	// If max pooling, we will initialize the vector index part.
	if (this->layer_param_.superpixel_pooling_param().pool()
			== SuperpixelPoolingParameter_PoolMethod_MAX && top.size() == 1) {
		max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
				pooled_width_);
	}
	// If stochastic pooling, we will initialize the random index part.
	if (this->layer_param_.superpixel_pooling_param().pool()
			== SuperpixelPoolingParameter_PoolMethod_STOCHASTIC) {
		rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
				pooled_width_);
	}
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template<typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int spatial_dim = height_ * width_;
	const int pooled_dim = pooled_height_ * pooled_width_;
	const int top_count = top[0]->count();
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more code.
	switch (this->layer_param_.superpixel_pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		// Initialize
		caffe_set(top_count, Dtype(-FLT_MAX), top_data);
		// The main loop
		for (int n = 0; n < bottom[0]->num(); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int pd = 0; pd < pooled_dim; ++pd) {
					for (int sd = 0; sd < spatial_dim; ++sd) {
						if (bottom[1]->cpu_data()[sd] == pd
								&& bottom_data[sd] > top_data[pd]) {
							top_data[pd] = bottom_data[sd];
						}
					}
				}

				// compute offset
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		for (int i = 0; i < top_count; ++i) {
			top_data[i] = 0;
		}
		// The main loop
		for (int n = 0; n < bottom[0]->num(); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int pd = 0; pd < pooled_dim; ++pd) {
					int pool_size = 0;
					for (int sd = 0; sd < spatial_dim; ++sd) {
						if (bottom[1]->cpu_data()[sd] == pd) {
							top_data[pd] += bottom_data[sd];
							pool_size++;
						}
					}
					top_data[pd] /= pool_size;
				}

				// compute offset
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}

template<typename Dtype>
void SuperpixelPoolingLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int spatial_dim = height_ * width_;
	const int pooled_dim = pooled_height_ * pooled_width_;
	// Different pooling methods. We explicitly do the switch outside the for
	// loop to save time, although this results in more codes.
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	switch (this->layer_param_.superpixel_pooling_param().pool()) {
	case PoolingParameter_PoolMethod_MAX:
		// The main loop
		for (int n = 0; n < top[0]->num(); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int pd = 0; pd < pooled_dim; ++pd) {
					for (int sd = 0; sd < spatial_dim; ++sd) {
						if (bottom[1]->cpu_data()[sd] == pd) {
							bottom_diff[sd] += top_diff[pd];
						}
					}
				}

				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_AVE:
		// The main loop
		for (int n = 0; n < top[0]->num(); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int pd = 0; pd < pooled_dim; ++pd) {
					int pool_size = 0;
					for (int sd = 0; sd < spatial_dim; ++sd) {
						if (bottom[1]->cpu_data()[sd] == pd) {
							pool_size++;
						}
					}
					for (int sd = 0; sd < spatial_dim; ++sd) {
						if (bottom[1]->cpu_data()[sd] == pd) {
							bottom_diff[sd] += top_diff[pd] / pool_size;
						}
					}
				}

				// offset
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
			}
		}
		break;
	case PoolingParameter_PoolMethod_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}

#ifdef CPU_ONLY
STUB_GPU(SuperpixelPoolingLayer);
#endif

INSTANTIATE_CLASS(SuperpixelPoolingLayer);
REGISTER_LAYER_CLASS(SuperpixelPooling);

}  // namespace caffe

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


// TODO: check we should clear confusion_matrix somewhere!

int n0 = 0;
int n1 = 0;
int n2 = 0;
namespace caffe {

template <typename Dtype>
void SegAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  confusion_matrix_.clear();
  confusion_matrix_.resize(bottom[0]->channels());
  SegAccuracyParameter seg_accuracy_param = this->layer_param_.seg_accuracy_param();
  for (int c = 0; c < seg_accuracy_param.ignore_label_size(); ++c){
    ignore_label_.insert(seg_accuracy_param.ignore_label(c));
  }
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(1, bottom[0]->channels())
      << "top_k must be less than or equal to the number of channels (classes).";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1)
    << "The label should have one channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data should have the same width as label.";
 
  top[0]->Reshape(1, 1, 1, 6);
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  //int dim = bottom[0]->count() / bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  Blob<Dtype> nii(1, channels, 1, 1);
  memset(nii.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
  Blob<Dtype> nji(1, channels, 1, 1);
  memset(nji.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
  Blob<Dtype> ti(1, channels, 1, 1);
  memset(ti.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
  Blob<Dtype> pl(1, channels, 1, 1);
  memset(pl.mutable_cpu_data(), 0, sizeof(Dtype) * channels);

  int data_index, label_index;

  int top_k = 1;  // only support for top_k = 1

  // remove old predictions if reset() flag is true
  //if (this->layer_param_.seg_accuracy_param().reset()) {
  //  confusion_matrix_.clear();
  //}

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
	// Top-k accuracy
	std::vector<std::pair<Dtype, int> > bottom_data_vector;

	for (int c = 0; c < channels; ++c) {
	  data_index = (c * height + h) * width + w;
	  bottom_data_vector.push_back(std::make_pair(bottom_data[data_index], c));
	}

	std::partial_sort(
	  bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
	  bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

	// check if true label is in top k predictions
	label_index = h * width + w;
	const int gt_label = static_cast<int>(bottom_label[label_index]);

	if (ignore_label_.count(gt_label) != 0) {
	  // ignore the pixel with this gt_label
	  continue;
	} else if (gt_label >= 0 && gt_label < channels) {
	  // current position is not "255", indicating ambiguous position
	  confusion_matrix_.accumulate(gt_label, bottom_data_vector[0].second);

	  pl.mutable_cpu_data()[bottom_data_vector[0].second]++;
	  if (bottom_data_vector[0].second == gt_label) {
	    nii.mutable_cpu_data()[gt_label]++;
	  } else {
	    nji.mutable_cpu_data()[gt_label]++;
	  }
	  ti.mutable_cpu_data()[gt_label]++;

	} else {
	  LOG(FATAL) << "Unexpected label " << gt_label << ". num: " << i 
              << ". row: " << h << ". col: " << w;
        }

        

      }
    }
    bottom_data  += bottom[0]->offset(1);
    bottom_label += bottom[1]->offset(1);
  }

  // we report all the resuls
  top[0]->mutable_cpu_data()[0] = (Dtype)confusion_matrix_.accuracy();
  top[0]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.avgRecall(false);
  top[0]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.avgJaccard();
  if (ti.cpu_data()[0] == 0) {
    top[0]->mutable_cpu_data()[3] = 0;
  } else {
    top[0]->mutable_cpu_data()[3] = (Dtype)nii.cpu_data()[0]/ti.cpu_data()[0];
    n0++;
  }
  if (ti.cpu_data()[1] == 0) {
    top[0]->mutable_cpu_data()[4] = 0;
  } else {
    top[0]->mutable_cpu_data()[4] = (Dtype)nii.cpu_data()[1]/ti.cpu_data()[1];
    n1++;
  }
  if (ti.cpu_data()[2] == 0) {
    top[0]->mutable_cpu_data()[5] = 0;
  } else {
    top[0]->mutable_cpu_data()[5] = (Dtype)nii.cpu_data()[2]/ti.cpu_data()[2];
    n2++;
  }
  LOG(INFO) << n0 << " " << n1 << " " << n2 ;
  /*
  Dtype result;

  switch (this->layer_param_.seg_accuracy_param().metric()) {
  case SegAccuracyParameter_AccuracyMetric_PixelAccuracy:
    result = (Dtype)confusion_matrix_.accuracy();
    break;
  case SegAccuracyParameter_AccuracyMetric_ClassAccuracy:
    result = (Dtype)confusion_matrix_.avgRecall();
    break;
  case SegAccuracyParameter_AccuracyMetric_PixelIOU:
    result = (Dtype)confusion_matrix_.avgJaccard();
    break;
  default:
    LOG(FATAL) << "Unknown Segment accuracy metric.";
  }
    
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = result;
  // Accuracy layer should not be used as a loss function.
  */
}

INSTANTIATE_CLASS(SegAccuracyLayer);
REGISTER_LAYER_CLASS(SegAccuracy);
}  // namespace caffe

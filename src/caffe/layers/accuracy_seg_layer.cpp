#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using namespace std;
namespace caffe {


template<typename Dtype>
void AccuracySegLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	/*const int channels = bottom[0]->channels();
	 for (int i = 0; i < channels; i++) {
	 num.push_back(0);
	 gnum.push_back(0);
	 }*/

	iter_ = 0;
	prefix_ = this->layer_param_.accuracy_seg_param().prefix();
	period_ = this->layer_param_.accuracy_seg_param().period();

	has_ignore_label_ =
			this->layer_param_.accuracy_seg_param().has_ignore_label();
	if (has_ignore_label_) {
		ignore_label_ = this->layer_param_.accuracy_seg_param().ignore_label();
	}

	if (this->layer_param_.accuracy_seg_param().has_source()) {
		std::ifstream infile(
				this->layer_param_.accuracy_seg_param().source().c_str());
		CHECK(infile.good()) << "Failed to open source file "
				<< this->layer_param_.accuracy_seg_param().source();
		const int strip = this->layer_param_.accuracy_seg_param().strip();
		CHECK_GE(strip, 0) << "Strip cannot be negative";
		string linestr;
		while (std::getline(infile, linestr)) {
			std::istringstream iss(linestr);
			string filename;
			iss >> filename;
			CHECK_GT(filename.size(), strip) << "Too much stripping";
			fnames_.push_back(filename.substr(0, filename.size() - strip));
		}
		LOG(INFO) << "Accuracy_seg will save a maximum of " << fnames_.size()
				<< " files.";
	}

}

template<typename Dtype>
void AccuracySegLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	/*LOG(INFO) << bottom[0]->count() << " " << bottom[0]->num() << " "
	 << bottom[0]->channels() << " " << bottom[0]->height() << " "
	 << bottom[0]->width();*/
	//CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
	//		<< "top_k must be less than or equal to the number of classes.";
	label_axis_ = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.accuracy_param().axis());
	outer_num_ = bottom[0]->count(0, label_axis_);
	inner_num_ = bottom[0]->count(label_axis_ + 1);

	/*ofstream ofile;
	 ofile.open("myfile.txt");
	 for (int i = 0; i < bottom[1]->height(); ++i) {
	 for (int j = 0; j < bottom[1]->width(); ++j) {
	 ofile << bottom[1]->cpu_data()[i * bottom[1]->width() + j] << " ";
	 }
	 ofile << "\n";
	 }
	 ofile.close();*/
	/*cv::Mat cv_label(bottom[1]->height(), bottom[1]->width(), CV_8U);
	 for (int i = 0; i < bottom[1]->height(); ++i) {
	 uchar * ptr = cv_label.ptr < uchar > (i);
	 for (int j = 0; j < bottom[1]->width(); ++j) {
	 ptr[j] = bottom[1]->cpu_data()[i * bottom[1]->width() + j];
	 }
	 }
	 CHECK(cv_label.data);
	 cv::Size dsize = cv::Size(bottom[0]->width(), bottom[0]->height());
	 cv::resize(cv_label, cv_label, dsize);
	 vector<int> label_shape(4);
	 label_shape[0] = bottom[0]->num();
	 label_shape[1] = 1;
	 label_shape[2] = bottom[0]->height();
	 label_shape[3] = bottom[0]->width();
	 bottom[1]->Reshape(label_shape);
	 Dtype* transformed_label = bottom[1]->mutable_cpu_data();

	 ofstream ofile;
	 ofile.open("myfile.txt");
	 for (int h = 0; h < bottom[0]->height(); ++h) {
	 uchar * ptr_label = cv_label.ptr < uchar > (h);
	 for (int w = 0; w < bottom[0]->width(); ++w) {
	 transformed_label[h * bottom[0]->width() + w] = ptr_label[w];
	 ofile << transformed_label[h * bottom[0]->width() + w] << " ";
	 LOG(INFO) << transformed_label[h * bottom[0]->width() + w];
	 }
	 ofile << "\n";
	 }
	 ofile.close();*/
	CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
			<< "Number of labels must match number of predictions; "
			<< "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
			<< "label count (number of labels) must be N*H*W, "
			<< "with integer values in {0, 1, ..., C-1}.";
	vector<int> top_shape(1);  // Accuracy is a scalar; 0 axes.
	top_shape[0] = 10;
	top[0]->Reshape(top_shape);

}

template<typename Dtype>
void AccuracySegLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->num();
	const int count = bottom[0]->count();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	const int spatial_dim = height * width;
	const int dim = count / num;
//exit(0);
	if (iter_ == fnames_.size()) {
		iter_ = 0;
	}
	if (iter_ % period_ == 0) {
		std::ostringstream oss;
		oss << prefix_;
		if (this->layer_param_.accuracy_seg_param().has_source()) {
			CHECK_LT(iter_, fnames_.size())
					<< "Test has run for more iterations than it was supposed to";
			oss << fnames_[iter_];
		} else {
			oss << "iter_" << iter_;
		}
		oss << "_parsing" << ".png";
		cv::Mat img_result(bottom[0]->height(), bottom[0]->width(), CV_8UC1);

		string label_savepath = oss.str().substr(0, oss.str().length() - 3)
				+ "txt"; //////
		LOG(INFO) << label_savepath;
		ofstream ofile;	/////////
		ofile.open(label_savepath.c_str());	///////////

		Blob<Dtype> intermediate_result(1, channels, 1, 1);
		memset(intermediate_result.mutable_cpu_data(), 0,
				sizeof(Dtype) * channels);
		Blob<Dtype> intermediate_result2(1, channels, 1, 1);
		Blob<Dtype> nii(1, channels, 1, 1);
		memset(nii.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
		Blob<Dtype> nji(1, channels, 1, 1);
		memset(nji.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
		Blob<Dtype> ti(1, channels, 1, 1);
		memset(ti.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
		Blob<Dtype> pl(1, channels, 1, 1);
		memset(pl.mutable_cpu_data(), 0, sizeof(Dtype) * channels);
		Dtype pixel_acc = 0;
		Dtype mean_acc = 0;
		Dtype mean_IU = 0;
		Dtype weighted_IU = 0;
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < spatial_dim; j++) {
				const int label_value = static_cast<int>(bottom[1]->cpu_data()[i
						* spatial_dim + j]);
				std::vector < std::pair<Dtype, int> > bottom_data_vector;

				/*LOG(INFO) << label_value;
				LOG(INFO) << bottom[0]->cpu_data()[i * dim + 0 * spatial_dim + j] << " " << bottom[0]->cpu_data()[i * dim + 1 * spatial_dim + j] << " " << bottom[0]->cpu_data()[i * dim + 2 * spatial_dim + j] << " ";*/

				for (int m = 0; m < channels; m++) {
					bottom_data_vector.push_back(
							std::make_pair(
									bottom[0]->cpu_data()[i * dim
											+ m * spatial_dim + j], m));
					// LOG(INFO) << m << " " << bottom[0]->cpu_data()[i * dim + m * spatial_dim + j];
				}
				std::partial_sort(bottom_data_vector.begin(),
						bottom_data_vector.begin() + channels,
						bottom_data_vector.end(),
						std::greater<std::pair<Dtype, int> >());
				int h = j / width;
				int w = j % width;
				//num[bottom_data_vector[0].second]++;
				//gnum[label_value]++;
				img_result.at < uchar > (h, w) = 255 / channels
						* bottom_data_vector[0].second;
				ofile << bottom_data_vector[0].second << " ";

				if (w == (width - 1)) {
					ofile << "\n";	//////
				}
				if (label_value != ignore_label_) {
					pl.mutable_cpu_data()[bottom_data_vector[0].second]++;
					if (bottom_data_vector[0].second == label_value) {
						nii.mutable_cpu_data()[label_value]++;
					} else {
						nji.mutable_cpu_data()[label_value]++;
					}
					ti.mutable_cpu_data()[label_value]++;
				}
			}
			cv::imwrite(oss.str().c_str(), img_result);
			ofile.close();	///////////

			cout << endl << "         right  wrong   total   predict" << endl;
			for (int hhh = 0; hhh < channels; hhh++) {
				cout << "class " << hhh << ": " << nii.cpu_data()[hhh] << "\t"
						<< nji.cpu_data()[hhh] << "\t" << ti.cpu_data()[hhh] << "\t" << pl.cpu_data()[hhh]
						<< endl;
			}
			cout << endl;

			//*************** Pixel accuracy ****************
			pixel_acc = caffe_cpu_asum(channels, nii.cpu_data()) * 1.0
					/ caffe_cpu_asum(channels, ti.cpu_data());

			//*************** Mean accuracy ****************
			for (int c = 0; c < channels; c++) {
				if (ti.cpu_data()[c] == 0) {
					intermediate_result.mutable_cpu_data()[c] = 0;
				} else {
					intermediate_result.mutable_cpu_data()[c] =
							nii.cpu_data()[c] * 1.0 / ti.cpu_data()[c];
				}
			}
			mean_acc = caffe_cpu_asum(channels, intermediate_result.cpu_data())
					* 1.0 / channels;

			//*************** Mean IU ****************
			caffe_add(channels, ti.cpu_data(), nji.cpu_data(),
					intermediate_result.mutable_cpu_data());
			caffe_sub(channels, intermediate_result.cpu_data(), nii.cpu_data(),
					intermediate_result.mutable_cpu_data());
			for (int c = 0; c < channels; c++) {
				if (intermediate_result.cpu_data()[c] == 0) {
					intermediate_result2.mutable_cpu_data()[c] = 0;
				} else {
					intermediate_result2.mutable_cpu_data()[c] =
							nii.cpu_data()[c] * 1.0
									/ intermediate_result.cpu_data()[c];
				}
			}
			mean_IU = caffe_cpu_asum(channels, intermediate_result2.cpu_data())
					* 1.0 / channels;

			//*************** Weighted IU ****************
			caffe_mul(channels, ti.cpu_data(), nii.cpu_data(),
					intermediate_result.mutable_cpu_data());
			for (int c = 0; c < channels; c++) {
				if (intermediate_result2.cpu_data()[c] == 0) {
					intermediate_result.mutable_cpu_data()[c] = 0;
				} else {
					intermediate_result.mutable_cpu_data()[c] =
							intermediate_result.cpu_data()[c] * 1.0
									/ intermediate_result2.cpu_data()[c];
				}
			}
			weighted_IU = caffe_cpu_asum(channels,
					intermediate_result.cpu_data()) * 1.0
					/ caffe_cpu_asum(channels, ti.cpu_data());
		}

		LOG(INFO) << "Pixel accuracy: " << pixel_acc / num;
		LOG(INFO) << "Mean accuracy:  " << mean_acc / num;
		LOG(INFO) << "Mean IU:        " << mean_IU / num;
		LOG(INFO) << "Weighted IU:    " << weighted_IU / num;

		top[0]->mutable_cpu_data()[0] = pixel_acc / num;
		top[0]->mutable_cpu_data()[1] = mean_acc / num;
		top[0]->mutable_cpu_data()[2] = mean_IU / num;
		top[0]->mutable_cpu_data()[3] = weighted_IU / num;
	}
	++iter_;

}

INSTANTIATE_CLASS(AccuracySegLayer);
REGISTER_LAYER_CLASS(AccuracySeg);

}  // namespace caffe

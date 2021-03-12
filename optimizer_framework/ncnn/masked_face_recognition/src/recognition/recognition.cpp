#include "recognition.h"


Recognize::Recognize(const std::string &model_param_path, const std::string &model_bin_path)
{
	std::string param_files = model_param_path;
	std::string bin_files = model_bin_path;
	#
	Recognet.load_param(param_files.c_str());
	Recognet.load_model(bin_files.c_str());
}

Recognize::~Recognize()
{
	Recognet.clear();
}

void Recognize::RecogNet(ncnn::Mat& img_, std::string &model_type)
{
	int embedding_size;
	ncnn::Extractor ex = Recognet.create_extractor();
	ex.set_num_threads(4);
	ex.set_light_mode(true);
	ncnn::Mat out;
	// check mode using face masked or face non-mask
	if(model_type=="mask")
	{
		embedding_size = 512;
		ex.input("input.1", img_);
		ex.extract("520_splitncnn_0", out);
	}
	else
	{
		embedding_size = 128;
		ex.input("data", img_);
		ex.extract("fc1", out);
	}
	feature_out.resize(embedding_size);
	for (int j = 0; j < embedding_size; j++)
	{
		feature_out[j] = out[j];
	}
}

void Recognize::start(const cv::Mat& img, std::vector<float>&feature, std::string &model_type)
{
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);
	RecogNet(ncnn_img, model_type);
	feature = feature_out;
}

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2)
{
	assert(v1.size() == v2.size());
	double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
	for (std::vector<double>::size_type i = 0; i != v1.size(); ++i)
	{
		ret += v1[i] * v2[i];
		mod1 += v1[i] * v1[i];
		mod2 += v2[i] * v2[i];
	}
	return (ret / sqrt(mod1) / sqrt(mod2) + 1) / 2.0;
}

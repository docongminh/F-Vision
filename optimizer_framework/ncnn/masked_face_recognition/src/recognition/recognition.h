#pragma once
#ifndef MOBILEFACENET_H_
#define MOBILEFACENET_H_
#include <string>
#include "net.h"
#include "opencv2/opencv.hpp"


class Recognize
{
	public:
		Recognize(const std::string &model_param_path, const std::string &model_bin_path);
		~Recognize();
		void start(const cv::Mat& img, std::vector<float>&feature, std::string &model_type);
	//
	private:
		void RecogNet(ncnn::Mat& img_, std::string &model_type);
		ncnn::Net Recognet;
		ncnn::Mat ncnn_img;
		std::vector<float> feature_out;
};

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2);


#endif // !MOBILEFACENET_H_
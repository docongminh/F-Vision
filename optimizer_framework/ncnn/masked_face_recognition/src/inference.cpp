#include <iostream>
#include <time.h>
//#include <opencv2/opencv.hpp>
#include "net.h"
#include "recognition.h"


int main()
{
	std::string model_type = "mask";
	char const *model_param_path, *model_bin_path;
	// check specify model type
	if(model_type=="mask")
	{
		model_param_path = "../models/mask/v1/masked_face.param";
		model_bin_path = "../models/mask/v1/masked_face.bin";
	}
	else
	{
		model_param_path = "../models/non-mask/mobilefacenet.param";
		model_bin_path = "../models/non-mask/mobilefacenet.bin";
	}
	Recognize recognize(model_param_path, model_bin_path);

	cv::Mat img1 = cv::imread("../images/minh/1.jpg", cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread("../images/minh/anchor.jpg", cv::IMREAD_COLOR);
	std::vector<float> feature1;
	std::vector<float> feature2;

	clock_t start_time = clock();
	recognize.start(img1, feature1, model_type);
	recognize.start(img2, feature2, model_type);
	double similar = calculSimilar(feature1, feature2);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

	std::cout << "time: " << total_time * 1000 << "ms" << std::endl;
	std::cout << "similarity is : " << similar << std::endl;

	return 0;
}

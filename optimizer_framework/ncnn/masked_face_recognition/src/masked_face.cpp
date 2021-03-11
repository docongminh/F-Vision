#include <iostream>
#include <time.h>
//#include <opencv2/opencv.hpp>
#include "net.h"
#include "masked_face_model.h"


int main()
{
	//char *model_path = "../models";
	char const *model_path = "../models/mask/v1";
	Recognize recognize(model_path);

	// cv::Mat img1 = cv::imread("../images/giang/1.jpg", cv::IMREAD_COLOR);
	// cv::Mat img2 = cv::imread("../images/giang/anchor.jpg", cv::IMREAD_COLOR);
	cv::Mat img1 = cv::imread("../images/minh/1.jpg", cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread("../images/minh/anchor.jpg", cv::IMREAD_COLOR);
	std::vector<float> feature1;
	std::vector<float> feature2;

	clock_t start_time = clock();
	recognize.start(img1, feature1);
	recognize.start(img2, feature2);
	double similar = calculSimilar(feature1, feature2);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

	std::cout << "time: " << total_time * 1000 << "ms" << std::endl;
	std::cout << "similarity is : " << similar << std::endl;

	return 0;
}

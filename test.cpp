#include <opencv2/opencv.hpp>
#include "LFFD.h"
#include<time.h>
// #include <chrono>
#include <filesystem>

// using namespace cv;
int main()
{
	// std::string const model_path = "../models";
	// std::string const image_file = "../images/t.png";
	char const *model_path = "../models";
	char const *image_file = "../images/t.png";

	LFFD lffd(model_path, 8, 32);
	cv::Mat images = cv::imread(image_file, cv::IMREAD_COLOR);
	std::vector<FaceInfo> face_info;

	ncnn::Mat inmat = ncnn::Mat::from_pixels(images.data, ncnn::Mat::PIXEL_BGR, images.cols, images.rows);
	clock_t start_time = clock();
	lffd.detect(inmat, face_info, 240, 320);
	clock_t end_time = clock();
	double total_time = (double) (end_time-start_time) / CLOCKS_PER_SEC;
	std::cout << "Time: "<< total_time * 1000 << "ms" << std::endl; 

	for (int i = 0; i < face_info.size(); i++)
	{
		auto face = face_info[i];
		cv::Point pt1(face.x1, face.y1);
		cv::Point pt2(face.x2, face.y2);
		cv::rectangle(images, pt1, pt2, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("../images/output.jpg", images);
	return 0;
}
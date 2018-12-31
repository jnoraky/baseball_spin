#include <vector>
#include <opencv2/opencv.hpp>

void process_data(const char *, const char *,std::vector<cv::Mat>&, std::vector<float>&,
   std::vector<cv::Point>&);
void process_data(int,std::vector<cv::Mat>&, std::vector<float>&,
   std::vector<cv::Point>&);

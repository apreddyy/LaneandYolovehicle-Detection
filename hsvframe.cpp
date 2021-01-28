//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::hsv_frame(const cv::cuda::GpuMat& src,
                               cv::cuda::GpuMat& dst) {
  cv::cuda::GpuMat hsv_frame, temp;
  cv::cuda::GpuMat channels_device[3];
  cv::cuda::cvtColor(src, hsv_frame, cv::COLOR_BGR2HSV);
  cv::cuda::split(hsv_frame, channels_device);
  cv::cuda::threshold(channels_device[0], channels_device[0], 0, 100,
                      cv::THRESH_BINARY);
  cv::cuda::threshold(channels_device[2], channels_device[1], 210, 255,
                      cv::THRESH_BINARY);
  cv::cuda::threshold(channels_device[2], channels_device[2], 200, 255,
                      cv::THRESH_BINARY);
  cv::cuda::merge(channels_device, 3, temp);
  cv::cuda::cvtColor(temp, dst, cv::COLOR_HSV2BGR);
  return;
}
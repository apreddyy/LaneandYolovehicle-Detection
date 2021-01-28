//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::binary_frame(const cv::cuda::GpuMat& src,
                                  cv::cuda::GpuMat& dst) {
  constexpr int threshold_value = 110;
  constexpr int max_value = 255;
  cv::cuda::threshold(src, dst, threshold_value, max_value, cv::THRESH_BINARY);
  return;
}
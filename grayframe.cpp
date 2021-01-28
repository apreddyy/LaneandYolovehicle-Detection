//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::gray_frame(const cv::cuda::GpuMat& src,
                                cv::cuda::GpuMat& dst) {
  cv::cuda::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
  return;
}
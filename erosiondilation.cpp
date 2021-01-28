//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::erode_dilate(const cv::cuda::GpuMat& src,
                                 cv::cuda::GpuMat& dst) {
  cv::cuda::GpuMat erode_out, dilate_out;
  constexpr int noise = 3;
  constexpr int dilate_const = 1;
  cv::Mat element_erosion = getStructuringElement(
      cv::MORPH_RECT, cv::Size(noise * 2 + 1, noise * 2 + 1));
  cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(
      cv::MORPH_ERODE, src.type(), element_erosion);
  erode->apply(src, erode_out);
  cv::Mat element_dilation = getStructuringElement(
      cv::MORPH_RECT, cv::Size(dilate_const * 2 + 1, dilate_const * 2 + 1));
  cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(
      cv::MORPH_DILATE, src.type(), element_dilation);
  dilateFilter->apply(erode_out, dst);
  return;
}
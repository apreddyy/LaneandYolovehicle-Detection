//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::sobel_frame(const cv::cuda::GpuMat& src,
                                 cv::cuda::GpuMat& dst) {
  cv::cuda::GpuMat sobelx, sobely, adwsobelx, adwsobely, gray_framea;
  gray_frame(src, gray_framea);
  cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSobelFilter(
      gray_framea.type(), CV_16S, 1, 0, 3, 1, cv::BORDER_DEFAULT);
  filter->apply(gray_framea, sobelx);
  cv::cuda::abs(sobelx, sobelx);
  sobelx.convertTo(dst, CV_8UC1);
  return;
}
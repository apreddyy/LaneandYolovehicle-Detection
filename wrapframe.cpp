//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::wrap_frame(const cv::cuda::GpuMat& src,
                                cv::cuda::GpuMat& dst,
                                const cv::Point2f* src_points,
                                const cv::Point2f* dst_points) {
  cv::Mat trans_points = getPerspectiveTransform(src_points, dst_points);
  cv::cuda::warpPerspective(src, dst, trans_points, src.size(),
                            cv::INTER_LINEAR);
  return;
}
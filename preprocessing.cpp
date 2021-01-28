//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::processing_pre(const cv::cuda::GpuMat& src,
                                    cv::cuda::GpuMat& resize,
                                    cv::cuda::GpuMat& dst) {
  cv::cuda::GpuMat resize_framea, gray_framea, binary_framea, birdview_framea,
      hsv_framea, sobel_frameout;
  if ((src.cols != resize_width) && (src.rows != resize_height)) {
    resize_frame(src, resize_framea, resize_height, resize_width);
    cv::cuda::remap(resize_framea, resize, gpu_mapx, gpu_mapy,
                    cv::INTER_LINEAR);
  } else {
    cv::cuda::remap(src, resize, gpu_mapx, gpu_mapy, cv::INTER_LINEAR);
  }
  wrap_frame(resize, birdview_framea, src_points, dst_points);
  sobel_frame(birdview_framea, sobel_frameout);
  hsv_frame(birdview_framea, hsv_framea);
  gray_frame(hsv_framea, gray_framea);
  binary_frame(gray_framea, binary_framea);
  cv::cuda::addWeighted(binary_framea, 0.9, sobel_frameout, 0.1, -1, dst);
  return;
}
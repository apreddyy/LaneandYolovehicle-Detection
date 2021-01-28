//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void left_point(const std::vector<int>& left_X, const std::vector<int>& main_Y,
                std::vector<cv::Point2i>& Pointleft) {
  int m = int(main_Y.size());
  for (int r = 0; r < m; r++) {
    Pointleft.push_back(cv::Point2i(left_X[r], main_Y[r]));
  }
}

void right_point(const std::vector<int>& right_X,
                 const std::vector<int>& main_Y,
                 std::vector<cv::Point2i>& Pointright) {
  int m = int(main_Y.size());
  for (int r = 0; r < m; r = r + 10) {
    int c = 359 - r;
    Pointright.push_back(cv::Point2i(right_X[c], main_Y[c]));
  }
  return;
}

void laneprocessing::processing_post(const cv::Mat& frame,
                                     const cv::cuda::GpuMat& src,
                                     cv::cuda::GpuMat& dst) {
  cv::cuda::GpuMat unwrap_framein, unwrap_frameout, cuda_frameout, dilate_out;

  std::vector<float> polyleft_in;
  std::vector<float> polyright_in;
  std::vector<int> leftx;
  std::vector<int> rightx;
  std::vector<int> main_y;
  std::vector<cv::Point2i> PointLeftRight;
  cv::Mat maskImage = cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0));
  std::vector<cv::Point2i> PointLeft;
  std::vector<cv::Point2i> PointRight;

  cuda_frameout.upload(frame);
  erode_dilate(cuda_frameout, dilate_out);
  video_frame(dilate_out, polyleft_in, polyright_in);
  if ((polyleft_in.size() == 0) || (polyright_in.size() == 0)) {
    set_leftcurvature(0);
    set_rightcurvature(0);
    set_centerdistance(0);
    goto endfunction;
  }
  curvature_sanitycheck(polyleft_in, polyright_in, leftx, rightx, main_y);

  left_point(leftx, main_y, PointLeft);
  right_point(rightx, main_y, PointRight);

  PointLeft.insert(PointLeft.end(), PointRight.begin(), PointRight.end());
  PointLeftRight = PointLeft;

  polylines(maskImage, PointLeft, false, cv::Scalar(0, 0, 255), 20, 150, 0);
  polylines(maskImage, PointRight, false, cv::Scalar(0, 0, 255), 20, 150, 0);

  const cv::Point* pts = (const cv::Point*)cv::Mat(PointLeftRight).data;
  int npts = cv::Mat(PointLeftRight).rows;
  fillPoly(maskImage, &pts, &npts, 1, cv::Scalar(0, 255, 0), 8);
  unwrap_framein.upload(maskImage);
  wrap_frame(unwrap_framein, unwrap_frameout, dst_points, src_points);
  cv::cuda::addWeighted(src, 1, unwrap_frameout, 0.5, -1, dst);
endfunction:
  dst = src;
  return;
}
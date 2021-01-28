//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneyolo::lane_detection() {
  cv::VideoCapture cap("opencv_resized.mp4");
  cv::String window_name = "Lane Display";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << std::endl;
  }
  cv::Mat frame, cpu_frame, display_frame;
  cv::cuda::GpuMat preprocess_frame, postprocess_frameout, resized_frame,
      process_frame;
  int64 tickmark = cv::getTickCount();
  while (1) {
    cap >> frame;
    if (frame.empty()) break;

    process_frame.upload(frame);
    processing_pre(process_frame, resized_frame, preprocess_frame);
    preprocess_frame.download(cpu_frame);
    processing_post(cpu_frame, process_frame, postprocess_frameout);
    postprocess_frameout.download(display_frame);
    cv::imshow(window_name, display_frame);
    if (cv::waitKey(10) >= 0) {
      set_leftcurvature(0);
      set_rightcurvature(0);
      set_centerdistance(0);
      break;
    }
    double fps = cv::getTickFrequency() / (cv::getTickCount() - tickmark);
    tickmark = cv::getTickCount();
    // std::cout << " THE FPS LANE: " << fps << std::endl;
    if (stop_requested()) {
      set_leftcurvature(0);
      set_rightcurvature(0);
      set_centerdistance(0);
      break;
    }
  }
  cap.release();
  cv::destroyAllWindows();
  std::cout << "Exiting Lane Process Thread" << std::endl;
  return;
}

float laneprocessing::get_leftcurvature(void) { return left_curverad; }

float laneprocessing::get_rightcurvature(void) { return right_curverad; }

float laneprocessing::get_centerdistance(void) { return center_dist; }

//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#ifndef LANEYOLO_HPP
#define LANEYOLO_HPP

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <conio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <thread>
#include <vector>

class threadstopper {
  std::promise<void> exitSignal;
  std::future<void> futureObj;

 public:
  threadstopper() : futureObj(exitSignal.get_future()) {}
  threadstopper(threadstopper&& obj)
      : exitSignal(std::move(obj.exitSignal)),
        futureObj(std::move(obj.futureObj)) {
    std::cout << "Move Constructor is called" << std::endl;
  }
  threadstopper& operator=(threadstopper&& obj) {
    std::cout << "Move Assignment is called" << std::endl;
    exitSignal = std::move(obj.exitSignal);
    futureObj = std::move(obj.futureObj);
    return *this;
  }
  // Task need to provide defination  for this function
  // It will be called by thread function
  virtual void lane_detection() = 0;
  virtual void yolo_preprocessing() = 0;
  virtual void yolo_detector() = 0;
  // Thread function to be executed by thread
  void operator()() {
    lane_detection();
    yolo_preprocessing();
    yolo_detector();
  }
  // Checks if thread is requested to stop
  bool stop_requested() {
    // checks if value in future object is available
    if (futureObj.wait_for(std::chrono::milliseconds(0)) ==
        std::future_status::timeout)
      return false;
    return true;
  }
  // Request the thread to stop by setting value in promise object
  void stop() { exitSignal.set_value(); }
};

class lastfit {
 public:
  // Saves good polyfit left lane.
  static std::vector<float> polyright_last;
  // Saves good polyfit right lane.
  static std::vector<float> polyleft_last;
};

class laneprocessing {
 private:
  cv::cuda::GpuMat gpu_mapx, gpu_mapy;
  // Sets the height for image processing.
  static constexpr int resize_height = 360;
  // Sets the width for image processing.
  static constexpr int resize_width = 640;
  // SRC points to convert wrap image.
  cv::Point2f src_points[4] = {cv::Point2f(290, 230), cv::Point2f(350, 230),
                               cv::Point2f(520, 340), cv::Point2f(130, 340)};
  // DST points to convert wrap image.
  cv::Point2f dst_points[4] = {cv::Point2f(130, 0), cv::Point2f(520, 0),
                               cv::Point2f(520, 360), cv::Point2f(130, 360)};
  // Distance from center of car to end lanes.
  float center_dist;
  // Left curvature.
  float left_curverad;
  // Right curvature.
  float right_curverad;
  // Performs polifit on cordinates to fit curvature.
  std::vector<float> polyfit_eigen(const std::vector<float>& xv,
                                   const std::vector<float>& yv, int order);
  // Performs polyval.
  std::vector<float> polyval_eigen(const std::vector<float>& oCoeff,
                                   const std::vector<float>& oX);
  // Convert RGB to Gray on GPU.
  void gray_frame(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
  // Convert Gray to Binary on GPU.
  void binary_frame(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
  // Convert RGB to HSV on GPU.
  void hsv_frame(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
  // Converts image to BirdEye view on GPU.
  void wrap_frame(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                  const cv::Point2f* src_points, const cv::Point2f* dst_points);
  // Converts RGB to Sobel on GPU.
  void sobel_frame(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
  // Dilates Image on GPU.
  void erode_dilate(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
  // Performs pre process on first frame.
  void initial_frame(const cv::cuda::GpuMat& src,
                     std::vector<float>& polyright_first,
                     std::vector<float>& polyleft_first);
  // Process the continous frames from Video.
  void video_frame(const cv::cuda::GpuMat& src,
                   std::vector<float>& polyleft_out,
                   std::vector<float>& polyright_out);
  // Checks the sanity of survature and lanes.
  void curvature_sanitycheck(std::vector<float>& polyleft_in,
                             std::vector<float>& polyright_in,
                             std::vector<int>& leftx, std::vector<int>& rightx,
                             std::vector<int>& main_y);
  // Gets the cordinates for lane pixel.
  void nonzero_pixelsinitial(const cv::cuda::GpuMat& src,
                             std::vector<float>& output_hx,
                             std::vector<float>& output_hy);
  // Gets the cordinate for lane pixel continous.
  void nonzero_pixelsnext(const cv::cuda::GpuMat& src,
                          std::vector<float>& loutput_hx,
                          std::vector<float>& loutput_hy,
                          std::vector<float>& routput_hx,
                          std::vector<float>& routput_hy);

 public:
  // Initilize intrinsicn, distCoeffsn to undisort frame.
  laneprocessing() : center_dist(0), left_curverad(0), right_curverad(0) {
    cv::Mat mapx, mapy;
    cv::Mat intrinsicn = cv::Mat(3, 3, CV_32FC1);
    cv::Mat distCoeffsn = cv::Mat(3, 3, CV_32FC1);
    cv::FileStorage fs2("calibration.yml", cv::FileStorage::READ);
    fs2["intrinsic"] >> intrinsicn;
    fs2["distCoeffs"] >> distCoeffsn;
    cv::initUndistortRectifyMap(intrinsicn, distCoeffsn,
                                cv::Mat::eye(3, 3, CV_32FC1), intrinsicn,
                                cv::Size(640, 360), CV_32FC1, mapx, mapy);
    gpu_mapx.upload(mapx);
    gpu_mapy.upload(mapy);
  }
  // Resize image on GPU.
  void resize_frame(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst,
                    int resize_height, int resize_width);
  // Performs pre process for image to find lanes.
  void processing_pre(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& resize,
                      cv::cuda::GpuMat& dst);
  // Finds lane pixel and calculate curvature and lines and performs post
  // process.
  void processing_post(const cv::Mat& frame, const cv::cuda::GpuMat& src,
                       cv::cuda::GpuMat& dst);
  // Returns Left Curvature.
  float get_leftcurvature(void);
  // Returns Right Curvature.
  float get_rightcurvature(void);
  // Returns Vehicle center distance from lanes center to end lanes.
  float get_centerdistance(void);
  // Set Left Curvature.
  void set_leftcurvature(float);
  // Set Right Curvature.
  void set_rightcurvature(float);
  // Set Vehicle center distance from lanes center to end lanes.
  void set_centerdistance(float);
};

class yoloprocessing {
 public:
  std::queue<cv::Mat> vecMat;
  torch::Tensor batch_Image;
  int paddingl;
  int paddingr;
  int paddingt;
  int paddingb;
  std::mutex lock_thread;
  std::condition_variable lock_cond;
  bool emptyFrame = false;
  torch::jit::script::Module model;
  float iou_thres;
  float conf_thres;
  int batchSize;
  // Initilize model from pt.
  yoloprocessing()
      : iou_thres(0.6),
        conf_thres(0.4),
        batchSize(1),
        paddingl(0),
        paddingr(0),
        paddingt(0),
        paddingb(280) {
    try {
      torch::NoGradGuard no_grad;
      model =
          torch::jit::load("best.torchscript.pt", torch::Device(torch::kCUDA));
    } catch (const c10::Error& e) {
      std::cout << e.what() << std::endl;
      std::cerr << "Error loading the model\n";
    }
  }
};

class laneyolo : public yoloprocessing,
                 public threadstopper,
                 public laneprocessing {
 public:
  void rectcv(const at::TensorAccessor<float, 2>& off_boxes,
              const at::TensorAccessor<float, 2>& detection_device,
              std::vector<cv::Rect>& offset_boxvec,
              std::vector<float>& score_vec);
  void lane_detection();
  void yolo_preprocessing();
  void yolo_detector();
  // Set batch size for detection.
  void set_batchsize(int);
  // Sets IOU threshold for detection.
  void set_iouthres(float);
  // Sets Confidance threshold for detection.
  void set_confthres(float);
  // Sets padding for image for detection.
  void set_padding(int lpadding, int rpadding, int tpadding, int bpadding);
};

#endif  // LANEYOLO_HPP
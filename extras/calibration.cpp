//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include <iostream>
#include <conio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

bool do_calib = true;

int main(int, const char* const[]) { 

  if (do_calib == true) {
    constexpr int numBoards = 17;
    constexpr int numCornersHor = 9;
    constexpr int numCornersVer = 6;
    constexpr int numSquares = numCornersHor * numCornersVer;
    cv::Size board_sz = cv::Size(numCornersHor, numCornersVer);
    cv::VideoCapture capture;
    capture.open("%02d.jpg");

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    std::vector<cv::Point2f> corners;
    int successes = 0;
    cv::Mat image, gray_image;

    capture >> image;
    int chessBoardFlags =
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    std::vector<cv::Point3f> obj;
    for (int j = 0; j < numSquares; j++)
      obj.push_back(cv::Point3f(float(j / numCornersHor),
                                float(j % numCornersHor), 0.0f));

    while (successes < numBoards) {
      cvtColor(image, gray_image, cv::COLOR_BGR2RGB);

      bool found =
          findChessboardCorners(image, board_sz, corners, chessBoardFlags);

      if (found) {
        cornerSubPix(
            gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                             30, 0.1));
        drawChessboardCorners(gray_image, board_sz, corners, found);
      }

      //capture >> image;
      int key = cv::waitKey(1);

      if (found != 0) {
        image_points.push_back(corners);
        object_points.push_back(obj);
        std::cout << "Snap stored!" << std::endl;
        successes++;
        if (successes >= numBoards) break;
      }
    }

    //cv::VideoCapture capt;
    //capt.open("%02d.jpg");
    //capt >> image;
    cv::Mat intrinsic = cv::Mat(3, 3, CV_32FC1);
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;

    calibrateCamera(object_points, image_points, image.size(), intrinsic,
                    distCoeffs, rvecs, tvecs);

    cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
    fs << "intrinsic" << intrinsic;
    fs << "distCoeffs" << distCoeffs;
    fs.release();
  }
  return 0;
}

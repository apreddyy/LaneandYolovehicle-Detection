//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

void laneprocessing::nonzero_pixelsinitial(const cv::cuda::GpuMat& src,
                                           std::vector<float>& output_hx,
                                           std::vector<float>& output_hy) {
  cv::Mat frame;
  src.download(frame);
  std::vector<cv::Point2f> locations;
  cv::findNonZero(frame, locations);
  for (auto i = 0; i < locations.size(); i++) {
    output_hx.push_back(locations[i].x);
    output_hy.push_back(locations[i].y);
  }
  return;
}

void laneprocessing::nonzero_pixelsnext(const cv::cuda::GpuMat& src,
                                        std::vector<float>& loutput_hx,
                                        std::vector<float>& loutput_hy,
                                        std::vector<float>& routput_hx,
                                        std::vector<float>& routput_hy) {
  cv::Mat frame;
  src.download(frame);
  std::vector<cv::Point2f> locations;
  cv::findNonZero(frame, locations);
  constexpr int margin = 15;
  for (auto i = 0; i < locations.size(); i++) {
    auto left_idx =
        ((locations[i].x > (lastfit::polyleft_last[2] * pow(locations[i].y, 2) +
                            lastfit::polyleft_last[1] * locations[i].y +
                            lastfit::polyleft_last[0] - margin)) &
         (locations[i].x <
          (lastfit::polyleft_last[2] * (pow(locations[i].y, 2)) +
           lastfit::polyleft_last[1] * locations[i].y +
           lastfit::polyleft_last[0] + margin)));
    auto right_idx = ((locations[i].x >
                       (lastfit::polyright_last[2] * pow(locations[i].y, 2) +
                        lastfit::polyright_last[1] * locations[i].y +
                        lastfit::polyright_last[0] - margin)) &
                      (locations[i].x <
                       (lastfit::polyright_last[2] * (pow(locations[i].y, 2)) +
                        lastfit::polyright_last[1] * locations[i].y +
                        lastfit::polyright_last[0] + margin)));

    if (left_idx != 0) {
      loutput_hx.push_back(locations[i].x);
      loutput_hy.push_back(locations[i].y);
    }

    if (right_idx != 0) {
      routput_hx.push_back(locations[i].x);
      routput_hy.push_back(locations[i].y);
    }
  }
  return;
}
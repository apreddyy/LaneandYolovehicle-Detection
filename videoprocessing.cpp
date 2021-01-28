//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//#                                  Pramod Reddy                                     #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//

#include "laneyolo.hpp"

std::vector<float> lastfit::polyright_last;
std::vector<float> lastfit::polyleft_last;

void checkvec_size(std::size_t lx_size, std::size_t ly_size,
                   std::size_t rx_size, std::size_t ry_size) {
  if ((lx_size == 0) || (ly_size == 0) || (rx_size == 0) || (ry_size == 0)) {
    throw "The returned size of leftx, lefty, rightx, righty is 0";
  }
}

void laneprocessing::video_frame(const cv::cuda::GpuMat& src,
                                 std::vector<float>& polyleft_out,
                                 std::vector<float>& polyright_out) {
  if ((lastfit::polyleft_last.size() == 0) &&
      (0 == lastfit::polyright_last.size())) {
    initial_frame(src, polyright_out, polyleft_out);
    lastfit::polyright_last = polyright_out;
    lastfit::polyleft_last = polyleft_out;
  } else {
    std::vector<float> leftx;
    std::vector<float> lefty;
    std::vector<float> rightx;
    std::vector<float> righty;
    nonzero_pixelsnext(src, leftx, lefty, rightx, righty);

    try {
      checkvec_size(leftx.size(), lefty.size(), rightx.size(), righty.size());
    } catch (const char* msg) {
      std::cerr << msg << std::endl;
      goto endfunction;
    }

    polyright_out = polyfit_eigen(righty, rightx, 2);
    polyleft_out = polyfit_eigen(lefty, leftx, 2);
  }
endfunction:
  return;
}